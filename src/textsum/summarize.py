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

from textsum.utils import get_timestamp, postprocess_booksummary


class Summarizer:
    """
    Summarizer - a class that contains functions for summarizing text
    """

    def __init__(
        self,
        model_name_or_path: str = "pszemraj/long-t5-tglobal-base-16384-book-summary",
        use_cuda: bool = True,
        is_general_attention_model: bool = True,
        token_batch_length: int = 2048,
        batch_stride: int = 16,
        max_len_ratio: float = 0.25,
        **kwargs,
    ):
        """
        __init__ - initialize the Summarizer class

        :param str model_name_or_path: the name or path of the model to load, defaults to "pszemraj/long-t5-tglobal-base-16384-book-summary"
        :param bool use_cuda: whether to use cuda, defaults to True
        :param bool is_general_attention_model: whether the model is a general attention model, defaults to True
        :param int token_batch_length: the amount of tokens to process in a batch, defaults to 2048
        :param int batch_stride: the amount of tokens to stride the batch by, defaults to 16
        :param float max_len_ratio: the ratio of the token_batch_length to use as the max_length for the model, defaults to 0.25
        :param kwargs: additional keyword arguments to pass to the model as inference parameters
        """
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.logger.debug(f"loading model {model_name_or_path} to {self.device}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.logger.info(f"Loaded model {model_name_or_path} to {self.device}")
        self.is_general_attention_model = is_general_attention_model

        # set batch processing parameters
        self.token_batch_length = token_batch_length
        self.batch_stride = batch_stride
        self.max_len_ratio = max_len_ratio

        self.inference_params = {
            "min_length": 8,
            "max_length": int(token_batch_length * max_len_ratio),
            "no_repeat_ngram_size": 3,
            "encoder_no_repeat_ngram_size": 4,
            "repetition_penalty": 2.5,
            "num_beams": 4,
            "num_beam_groups": 1,
            "length_penalty": 0.8,
            "early_stopping": True,
            "do_sample": False,
        }

        for key, value in kwargs.items():
            if key in self.inference_params:
                self.inference_params[key] = value
            else:
                self.logger.warning(
                    f"{key} is not a supported inference parameter, ignoring"
                )

    def set_inference_params(self, new_params: dict):
        """update the inference parameters with new parameters"""
        for key, value in new_params.items():
            if key in self.inference_params:
                self.inference_params[key] = value
            else:
                self.logger.warning(
                    f"{key} is not a valid inference parameter, ignoring"
                )

    def get_inference_params(self):
        """get the inference parameters currently being used"""
        return self.inference_params

    def summarize_and_score(self, ids, mask, **kwargs):
        """
        summarize_and_score - given a batch of ids and a mask, return a summary and a score for the summary

        Args:
            ids (): the batch of ids
            mask (): the attention mask for the batch

        Returns:
            str: the summary of the batch
        """

        ids = ids[None, :]
        mask = mask[None, :]

        input_ids = ids.to("cuda") if torch.cuda.is_available() else ids
        attention_mask = mask.to("cuda") if torch.cuda.is_available() else mask

        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1

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
        score = round(summary_pred_ids.sequences_scores.cpu().numpy()[0], 4)

        return summary, score

    def summarize_via_tokenbatches(
        self,
        input_text: str,
        batch_length=None,
        batch_stride=None,
        **kwargs,
    ):
        """
        summarize_via_tokenbatches - a function that takes a string and returns a summary

        Args:
            input_text (str): the text to summarize
            batch_length (int, optional): overrides the default batch length. Defaults to None.
            batch_stride (int, optional): overrides the default batch stride. Defaults to None.

        Returns:
            str: the summary
        """

        logger = logging.getLogger(__name__)
        # log all input parameters
        if batch_length < 512 and batch_length is not None:
            batch_length = 512
            logger.warning("WARNING: batch_length was set to 512")
        logger.debug(
            f"batch_length: {batch_length} batch_stride: {batch_stride}, kwargs: {kwargs}"
        )
        # if received kwargs, update inference params
        if kwargs:
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
        pbar = tqdm(total=len(in_id_arr))

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
            logger.debug(f"\n\t{result[0]}\nScore:\t{score}")
            pbar.update()

        pbar.close()

        return gen_summaries

    def process_output(
        self,
        summary_data: dict,
        target_file: str or Path,
        postprocess: bool = True,
        custom_phrases: list = None,
        save_scores: bool = True,
        return_string: bool = False,
    ) -> None:
        """
        process_output - a function that takes the output of summarize_via_tokenbatches and saves it to a file after postprocessing

        :param dict summary_data: output of summarize_via_tokenbatches containing the summary and score for each batch
        :param str or Path target_file: the file to save the summary to
        :param bool postprocess: whether to postprocess the summary, defaults to True
        :param list custom_phrases: a list of custom phrases to use in postprocessing, defaults to None
        :param bool save_scores: whether to save the scores for each batch, defaults to True
        :param bool return_string: whether to return the summary as a string, defaults to False
        """

        target_file = Path(target_file).resolve()
        if target_file.exists():
            warnings.warn(f"File {target_file} exists, overwriting")

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
        with open(
            target_file,
            "w",
        ) as fo:

            fo.writelines(full_summary)

        if save_scores:
            with open(
                target_file,
                "a",
            ) as fo:

                fo.write("\n" * 3)
                fo.write(f"\n\nSection Scores for {target_file.stem}:\n")
                fo.writelines(scores_text)
                fo.write("\n\n---\n")

        logging.info(f"Saved summary to {target_file.resolve()}")

    def summarize_text_file(
        self,
        file_path: str or Path,
        output_dir: str or Path = None,
        batch_length=None,
        batch_stride=None,
        lowercase: bool = False,
        **kwargs,
    ) -> Path:
        """
        summarize_text_file - a function that takes a text file and returns a summary

        :param strorPath file_path: _description_
        :param strorPath output_dir: _description_, defaults to None
        :param bool lowercase: _description_, defaults to False

        :return Path: the path to the summary file
        """

        logger = logging.getLogger(__name__)

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

        self.process_output(
            gen_summaries,
            output_file,
        )

        return output_file

    def save_params(
        self,
        output_dir: str or Path = None,
        hf_tag: str = None,
        verbose: bool = False,
    ) -> None:
        """
        save_params - save the parameters of the run to a json file

        :param dict params: parameters to save
        :param str or Path output_dir: directory to save the parameters to
        :param str hf_tag: the model tag on huggingface
        :param bool verbose: whether to log the parameters

        :return: None
        """
        output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        session_settings = self.get_inference_params()
        session_settings["huggingface-model-tag"] = "" if hf_tag is None else hf_tag
        session_settings["date-run"] = get_timestamp()

        metadata_path = output_dir / "summarization-parameters.json"
        logging.info(f"Saving parameters to {metadata_path}")
        with open(metadata_path, "w") as write_file:
            json.dump(session_settings, write_file)

        logging.debug(f"Saved parameters to {metadata_path}")
        if verbose:
            # log the parameters
            logging.info(f"parameters: {session_settings}")
