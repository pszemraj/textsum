"""
summarize.py - a module that contains functions for summarizing text
"""
import json
import logging
import warnings
from pathlib import Path

import torch
from cleantext import clean
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textsum.utils import (
    check_bitsandbytes_available,
    get_timestamp,
    postprocess_booksummary,
)


class Summarizer:
    """
    Summarizer - a class that contains functions for summarizing text with a transformers model
    """

    def __init__(
        self,
        model_name_or_path: str = "pszemraj/long-t5-tglobal-base-16384-book-summary",
        use_cuda: bool = True,
        is_general_attention_model: bool = True,
        token_batch_length: int = 2048,
        batch_stride: int = 16,
        max_length_ratio: float = 0.25,
        load_in_8bit=False,
        **kwargs,
    ):
        """
        __init__ - initialize the Summarizer class

        :param str model_name_or_path: the name or path of the model to load, defaults to "pszemraj/long-t5-tglobal-base-16384-book-summary"
        :param bool use_cuda: whether to use cuda, defaults to True
        :param bool is_general_attention_model: whether the model is a general attention model, defaults to True
        :param int token_batch_length: the amount of tokens to process in a batch, defaults to 2048
        :param int batch_stride: the amount of tokens to stride the batch by, defaults to 16
        :param float max_length_ratio: the ratio of the token_batch_length to use as the max_length for the model, defaults to 0.25
        :param bool load_in_8bit: whether to load the model in 8bit precision (LLM.int8), defaults to False
        :param kwargs: additional keyword arguments to pass to the model as inference parameters
        """
        self.logger = logging.getLogger(__name__)

        self.model_name_or_path = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.logger.debug(f"loading model {model_name_or_path} to {self.device}")

        if load_in_8bit:
            logging.info("Loading model in 8-bit precision")

            if not check_bitsandbytes_available():
                raise ImportError(
                    "You must install bitsandbytes to load the model in 8-bit precision. Please run `pip install bitsandbytes` or `pip install textsum[8bit]`"
                )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                load_in_8bit=load_in_8bit,
                device_map="auto",
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
            ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.is_general_attention_model = (
            is_general_attention_model  # TODO: add a check later
        )

        self.logger.info(f"Loaded model {model_name_or_path} to {self.device}")

        # set batch processing parameters
        self.token_batch_length = token_batch_length
        self.batch_stride = batch_stride
        self.max_len_ratio = max_length_ratio

        self.settable_inference_params = [
            "min_length",
            "max_length",
            "no_repeat_ngram_size",
            "encoder_no_repeat_ngram_size",
            "repetition_penalty",
            "num_beams",
            "num_beam_groups",
            "length_penalty",
            "early_stopping",
            "do_sample",
        ]  # list of inference parameters that can be set
        self.inference_params = {
            "min_length": 8,
            "max_length": int(token_batch_length * max_length_ratio),
            "no_repeat_ngram_size": 3,
            "encoder_no_repeat_ngram_size": 4,
            "repetition_penalty": 2.5,
            "num_beams": 4,
            "num_beam_groups": 1,
            "length_penalty": 0.8,
            "early_stopping": True,
            "do_sample": False,
        }  # default inference parameters

        for key, value in kwargs.items():
            if key in self.settable_inference_params:
                self.inference_params[key] = value
            else:
                self.logger.warning(
                    f"{key} is not a supported inference parameter, ignoring"
                )

    def set_inference_params(
        self,
        new_params: dict = None,
        config_file: str or Path = None,
        config_metadata_id: str = "META_",
    ):
        """
        set_inference_params - update the inference parameters to use when summarizing text

        :param dict new_params: a dictionary of new inference parameters to use, defaults to None
        :param str or Path config_file: a path to a json file containing inference parameters, defaults to None

        NOTE: if both new_params and config_file are provided, entries in the config_file will overwrite entries in new_params if they have the same key
        """

        assert (
            new_params or config_file
        ), "must provide new_params or config_file to set inference parameters"

        new_params = new_params or {}
        # load from config file if provided
        if config_file:
            with open(config_file, "r") as f:
                config_params = json.load(f)
            config_params = {
                k: v
                for k, v in config_params.items()
                if k in self.settable_inference_params
            }  # remove key:value pairs that start with config_metadata_id
            new_params.update(config_params)
            self.logger.info(f"loaded inference parameters from {config_file}")
            self.logger.debug(f"inference parameters: {new_params}")

        for key, value in new_params.items():
            if key in self.settable_inference_params:
                self.inference_params[key] = value
            else:
                self.logger.warning(
                    f"{key} is not a valid inference parameter, ignoring"
                )

    def get_inference_params(self):
        """get the inference parameters currently being used"""
        return self.inference_params

    def update_loglevel(self, loglevel: int = logging.INFO):
        """update the loglevel of the logger"""
        self.logger.setLevel(loglevel)

    def summarize_and_score(self, ids, mask, **kwargs):
        """
        summarize_and_score - summarize a batch of text and return the summary and output scores

        :param ids: the token ids of the tokenized batch to summarize
        :param mask: the attention mask of the tokenized batch to summarize
        :return tuple: a tuple containing the summary and output scores
        """

        ids = ids[None, :]
        mask = mask[None, :]

        input_ids = ids.to("cuda") if torch.cuda.is_available() else ids
        attention_mask = mask.to("cuda") if torch.cuda.is_available() else mask

        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1

        self.logger.debug(
            f"generating summary for batch of size {input_ids.shape} with {kwargs}"
        )
        if self.is_general_attention_model:
            summary_pred_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )
        else:
            # this is for LED etc.
            summary_pred_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )
        summary = self.tokenizer.batch_decode(
            summary_pred_ids.sequences,
            skip_special_tokens=True,
            remove_invalid_values=True,
        )
        self.logger.debug(f"summary: {summary}")
        score = round(summary_pred_ids.sequences_scores.cpu().numpy()[0], 4)

        return summary, score

    def summarize_via_tokenbatches(
        self,
        input_text: str,
        batch_length: int = None,
        batch_stride: int = None,
        **kwargs,
    ):
        """
        summarize_via_tokenbatches - given a string of text, split it into batches of tokens and summarize each batch

        :param str input_text: the text to summarize
        :param int batch_length: number of tokens to include in each input batch, default None (self.token_batch_length)
        :param int batch_stride: number of tokens to stride between batches, default None (self.token_batch_stride)
        :return: a list of summaries, a list of scores, and a list of the input text for each batch
        """

        # log all input parameters
        if batch_length and batch_length < 512:
            self.logger.warning(
                "WARNING: entered batch_length was too low at {batch_length}, resetting to 512"
            )
            batch_length = 512

        self.logger.debug(
            f"batch_length: {batch_length} batch_stride: {batch_stride}, kwargs: {kwargs}"
        )
        if kwargs:
            # if received kwargs, update inference params
            self.set_inference_params(**kwargs)

        params = self.get_inference_params()

        encoded_input = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=batch_length or self.token_batch_length,
            stride=batch_stride or self.batch_stride,
            return_overflowing_tokens=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask

        gen_summaries = []
        pbar = tqdm(total=len(in_id_arr), desc="Generating Summaries")

        for _id, _mask in zip(in_id_arr, att_arr):

            result, score = self.summarize_and_score(
                ids=_id,
                mask=_mask,
                **params,
            )
            score = round(float(score), 4)
            _sum = {
                "input_tokens": _id,
                "summary": result,
                "summary_score": score,
            }
            gen_summaries.append(_sum)
            self.logger.debug(f"\n\t{result[0]}\nScore:\t{score}")
            pbar.update()

        pbar.close()

        return gen_summaries

    def save_summary(
        self,
        summary_data: dict,
        target_file: str or Path = None,
        postprocess: bool = True,
        custom_phrases: list = None,
        save_scores: bool = True,
        return_string: bool = False,
    ):
        """
        save_summary - a function that takes the output of summarize_via_tokenbatches and saves it to a file after postprocessing

        :param dict summary_data: output of summarize_via_tokenbatches containing the summary and score for each batch
        :param str or Path target_file: the file to save the summary to, defaults to None
        :param bool postprocess: whether to postprocess the summary, defaults to True
        :param list custom_phrases: a list of custom phrases to use in postprocessing, defaults to None
        :param bool save_scores: whether to save the scores for each batch, defaults to True
        :param bool return_string: whether to return the summary as a string, defaults to False

        :return: None or str if return_string is True
        """
        assert (
            target_file or return_string
        ), "Must specify a target file or return_string=True"

        if postprocess:
            sum_text = [
                postprocess_booksummary(
                    s["summary"][0],
                    custom_phrases=custom_phrases,
                )
                for s in summary_data
            ]
        else:
            sum_text = [s["summary"][0] for s in summary_data]

        sum_scores = [f"\n - {round(s['summary_score'],4)}" for s in summary_data]
        scores_text = "\n".join(sum_scores)
        full_summary = "\n\t".join(sum_text)

        if return_string:
            return full_summary

        target_file = Path(target_file)
        if not target_file.parent.exists():
            logging.info(f"Creating directory {target_file.parent}")
            target_file.parent.mkdir(parents=True)
        if target_file.exists():
            warnings.warn(f"File {target_file} exists, overwriting")

        with open(
            target_file,
            "w",
            encoding="utf-8",
            errors="ignore",
        ) as fo:

            fo.writelines(full_summary)

        if save_scores:
            with open(
                target_file,
                "a",
                encoding="utf-8",
                errors="ignore",
            ) as fo:

                fo.write("\n" * 3)
                fo.write(f"\n\nSection Scores for {target_file.stem}:\n")
                fo.writelines(scores_text)
                fo.write("\n\n---\n")

        self.logger.info(f"Saved summary to {target_file.resolve()}")

    def summarize_string(
        self,
        input_text: str,
        batch_length: int = None,
        batch_stride: int = None,
        **kwargs,
    ) -> str:
        """
        summarize_string - generate a summary for a string of text

        :param str input_text: the text to summarize
        :param int batch_length: number of tokens to use in each batch, defaults to None (self.token_batch_length)
        :param int batch_stride: number of tokens to stride between batches, defaults to None (self.batch_stride)
        :return str: the summary
        """

        logger = logging.getLogger(__name__)
        # log all input parameters
        if batch_length and batch_length < 512:
            logger.warning(
                "WARNING: entered batch_length was too low at {batch_length}, resetting to 512"
            )
            batch_length = 512

        logger.debug(
            f"batch_length: {batch_length} batch_stride: {batch_stride}, kwargs: {kwargs}"
        )

        gen_summaries = self.summarize_via_tokenbatches(
            input_text,
            batch_length=batch_length,
            batch_stride=batch_stride,
            **kwargs,
        )

        return self.save_summary(summary_data=gen_summaries, return_string=True)

    def summarize_file(
        self,
        file_path: str or Path,
        output_dir: str or Path = None,
        batch_length=None,
        batch_stride=None,
        lowercase: bool = False,
        **kwargs,
    ) -> Path:
        """
        summarize_file - summarize a text file and save the summary to a file

        :param str or Path file_path: the path to the text file
        :param str or Path output_dir: the directory to save the summary to, defaults to None (current working directory)
        :param int batch_length: number of tokens to use in each batch, defaults to None (self.token_batch_length)
        :param int batch_stride: number of tokens to stride between batches, defaults to None (self.batch_stride)
        :param bool lowercase: whether to lowercase the text prior to summarization, defaults to False

        :return Path: the path to the summary file
        """

        file_path = Path(file_path)
        output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        output_file = output_dir / f"{file_path.stem}_summary.txt"

        with open(file_path, "r") as f:
            text = clean(f.read(), lower=lowercase)

        gen_summaries = self.summarize_via_tokenbatches(
            text,
            batch_length=batch_length,
            batch_stride=batch_stride,
            **kwargs,
        )

        self.save_summary(
            gen_summaries,
            output_file,
        )

        return output_file

    def save_params(
        self,
        output_path: str or Path = None,
        hf_tag: str = None,
        verbose: bool = False,
    ) -> None:
        """
        save_params - save the parameters of the run to a json file

        :param dict params: parameters to save
        :param str or Path output_path: directory or filepath to save the parameters to
        :param str hf_tag: the model tag on huggingface (will be used instead of self.model_name_or_path)
        :param bool verbose: whether to log the parameters

        :return: None
        """
        output_path = Path(output_path) if output_path is not None else Path.cwd()
        metadata_path = (
            output_path / "summarization_parameters.json"
            if output_path.is_dir()
            else output_path
        )  # if output_path is a file, use that, otherwise use the default name

        exported_params = self.get_inference_params().copy()
        exported_params["META_huggingface_model"] = (
            self.model_name_or_path if hf_tag is None else hf_tag
        )
        exported_params["META_date"] = get_timestamp()

        self.logger.info(f"Saving parameters to {metadata_path}")
        with open(metadata_path, "w") as write_file:
            json.dump(exported_params, write_file, indent=4)

        logging.debug(f"Saved parameters to {metadata_path}")
        if verbose:
            self.logger.info(f"parameters: {exported_params}")
            print(f"saved parameters to {metadata_path}")
