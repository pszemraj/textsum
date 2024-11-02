"""
summarize.py - a module that contains functions for summarizing text
"""

import json
import logging
import pprint as pp
import sys
import warnings
from pathlib import Path
from typing import Union

import torch
from cleantext import clean
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import textsum
from textsum.utils import (
    check_bitsandbytes_available,
    get_timestamp,
    postprocess_booksummary,
    validate_pytorch2,
)


class Summarizer:
    """
    Summarizer - utility class for summarizing long text using a pretrained text2text model
    """

    settable_inference_params = [
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

    def __init__(
        self,
        model_name_or_path: str = "BEE-spoke-data/pegasus-x-base-synthsumm_open-16k",
        use_cuda: bool = True,
        is_general_attention_model: bool = True,
        token_batch_length: int = 4096,
        batch_stride: int = 16,
        max_length_ratio: float = 0.25,
        load_in_8bit: bool = False,
        compile_model: bool = False,
        optimum_onnx: bool = False,
        force_cache: bool = False,
        disable_progress_bar: bool = False,
        **kwargs,
    ):
        f"""
        __init__ - initialize the Summarizer class

        :param str model_name_or_path: name or path of the model to load, defaults to "BEE-spoke-data/pegasus-x-base-synthsumm_open-16k"
        :param bool use_cuda: whether to use cuda if available, defaults to True
        :param bool is_general_attention_model: whether the model is a general attention model, defaults to True
        :param int token_batch_length: number of tokens to split the text into for batch summaries, defaults to 4096
        :param int batch_stride: the amount of tokens to stride the batch by, defaults to 16
        :param float max_length_ratio: ratio of the token_batch_length to calculate max_length (of outputs), defaults to 0.25
        :param bool load_in_8bit: load the model in 8bit precision (LLM.int8), defaults to False
        :param bool compile_model: compile the model (pytorch 2.0+ only), defaults to False
        :param bool optimum_onnx: load the model in ONNX Runtime, defaults to False
        :param bool force_cache: force the model to use cache in generation, defaults to False
        :param bool disable_progress_bar: disable the per-document progress bar, defaults to False
        :param kwargs: additional keyword arguments to pass to the model as inference parameters, any of: {self.settable_inference_params}
        """
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.disable_progress_bar = disable_progress_bar
        self.force_cache = force_cache
        self.is_general_attention_model = is_general_attention_model
        self.model_name_or_path = model_name_or_path
        self.use_cuda = use_cuda
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
        elif optimum_onnx:
            import onnxruntime
            from optimum.onnxruntime import ORTModelForSeq2SeqLM

            if self.device == "cuda":
                self.logger.warning(
                    "ONNX runtime+cuda needs an additional package. manually install onnxruntime-gpu"
                )
            provider = (
                "CUDAExecutionProvider"
                if "GPU" in onnxruntime.get_device() and self.device == "cuda"
                else "CPUExecutionProvider"
            )
            self.logger.info(f"Loading model in ONNX Runtime to provider:\t{provider}")
            self.model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                provider=provider,
                export=not Path(self.model_name_or_path).is_dir(),
            )  # if a directory, already exported
            self.logger.warning(
                "ONNXruntime support is experimental, and functionality may vary per-model. "
                "Model outputs should be checked for accuracy"
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
            ).to(self.device)
            # device_map="auto" is not added for all models

        if compile_model:
            if validate_pytorch2() and sys.platform != "win32":
                self.logger.info("Compiling model")
                self.model = torch.compile(self.model)
            else:
                self.logger.warning(
                    "Unable to compile model. Please upgrade to PyTorch 2.0 and run on a non-Windows platform"
                )
        else:
            self.logger.debug("Not compiling model")

        if not optimum_onnx:
            self.model = self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.is_general_attention_model = (
            is_general_attention_model  # TODO: add a check later
        )

        self.logger.info(f"Loaded model {model_name_or_path} to {self.device}")

        if force_cache:
            self.logger.info("Forcing use_cache to True")
            self.model.config.use_cache = True
            self.logger.debug(
                f"model.config.use_cache: {pp.pformat(self.model.config.to_dict())}"
            )

        # set batch processing parameters
        self.token_batch_length = token_batch_length
        self.batch_stride = batch_stride
        self.max_len_ratio = max_length_ratio

        self.inference_params = {
            "min_length": 8,
            "max_length": int(token_batch_length * max_length_ratio),
            "no_repeat_ngram_size": 3,
            "encoder_no_repeat_ngram_size": 4,
            "repetition_penalty": 2.5,
            "num_beams": 4,
            "num_beam_groups": 1,
            "length_penalty": 1.0,
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

        self.config = {
            "model_name_or_path": model_name_or_path,
            "use_cuda": use_cuda,
            # "is_general_attention_model": is_general_attention_model, # TODO: validate later
            "token_batch_length": token_batch_length,
            "batch_stride": batch_stride,
            "max_length_ratio": max_length_ratio,
            "load_in_8bit": load_in_8bit,
            "compile_model": compile_model,
            "optimum_onnx": optimum_onnx,
            "device": self.device,
            "inference_params": self.inference_params,
            "textsum_version": textsum.__version__,
        }

    def __str__(self):
        return f"Summarizer({json.dumps(self.config)})"

    def __repr__(self):
        return self.__str__()

    def set_inference_params(
        self,
        new_params: dict = None,
        config_file: Union[str, Path] = None,
    ):
        """
        set_inference_params - update the inference parameters to use when summarizing text

        :param dict new_params: a dictionary of new inference parameters to use, defaults to None
        :param Union[str, Path] config_file: a path to a json file containing inference parameters, defaults to None

        NOTE: if both new_params and config_file are provided, entries in the config_file will overwrite entries in new_params if they have the same key
        """

        assert (
            new_params or config_file
        ), "must provide new_params or config_file to set inference parameters"

        new_params = new_params or {}
        # load from config file if provided
        if config_file:
            with open(config_file, "r", encoding="utf-8") as f:
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

    def print_config(self):
        """print the current configuration"""
        print(json.dumps(self.config, indent=2))

    def save_config(self, path: Union[str, Path] = "textsum_config.json"):
        """save the current configuration to a json file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def update_loglevel(self, loglevel: int = logging.INFO):
        """update the loglevel of the logger"""
        self.logger.setLevel(loglevel)

    def summarize_and_score(self, ids, mask, autocast_enabled: bool = False, **kwargs):
        """
        summarize_and_score - run inference on a batch of ids with the given attention mask

        :param ids: the token ids of the tokenized batch to summarize
        :param mask: the attention mask of the tokenized batch to summarize
        :param bool autocast_enabled: whether to use autocast for inference
        :return tuple: a tuple containing the summary and output scores
        """

        ids = ids[None, :]
        mask = mask[None, :]

        input_ids = ids.to("cuda") if torch.cuda.is_available() else ids
        attention_mask = mask.to("cuda") if torch.cuda.is_available() else mask

        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1

        self.logger.debug(f"gen. summary batch, size {input_ids.shape} with {kwargs}")
        with torch.autocast(device_type=self.device, enabled=autocast_enabled):
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
        min_batch_length: int = 512,
        pad_incomplete_batch: bool = True,
        disable_progress_bar: bool = None,
        **kwargs,
    ):
        """
        summarize_via_tokenbatches - given a string of text, split it into batches of tokens and summarize each batch

        :param str input_text: the text to summarize
        :param int batch_length: number of tokens to include in each input batch, default None (self.token_batch_length)
        :param int batch_stride: number of tokens to stride between batches, default None (self.token_batch_stride)
        :param int min_batch_length: minimum number of tokens in a batch, default 512
        :param bool pad_incomplete_batch: whether to pad the last batch to the length of the longest batch, default True
        :param bool disable_progress_bar: whether to disable the progress bar, default None
        :param kwargs: additional keyword arguments to pass to the summarize_and_score function

        :return: a list of summaries, a list of scores, and a list of the input text for each batch
        """

        batch_length = self.token_batch_length if batch_length is None else batch_length
        batch_stride = self.batch_stride if batch_stride is None else batch_stride
        disable_progress_bar = (
            self.disable_progress_bar
            if disable_progress_bar is None
            else disable_progress_bar
        )

        if batch_length < min_batch_length:
            self.logger.warning(
                f"batch_length must be at least {min_batch_length}. Setting batch_length to {min_batch_length}"
            )
            batch_length = min_batch_length

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
            max_length=batch_length,
            stride=batch_stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask

        gen_summaries = []
        pbar = tqdm(
            total=len(in_id_arr),
            desc="Generating Summaries",
            disable=disable_progress_bar,
        )
        for _id, _mask in zip(in_id_arr, att_arr):
            # If the batch is smaller than batch_length, pad it with the model's pad token
            if len(_id) < batch_length and pad_incomplete_batch:
                self.logger.debug(
                    f"padding batch of length {len(_id)} to {batch_length}"
                )
                pad_token = self.tokenizer.pad_token_id
                difference = batch_length - len(_id)
                _id = torch.cat([_id, torch.tensor([pad_token] * difference)])
                _mask = torch.cat([_mask, torch.tensor([0] * difference)])

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
        target_file: Union[str, Path] = None,
        postprocess: bool = True,
        batch_delimiter: str = "\n\n",
        custom_phrases: list = None,
        save_scores: bool = True,
        return_string: bool = False,
    ):
        """
        save_summary - a function that takes the output of summarize_via_tokenbatches and saves it to a file after postprocessing

        :param dict summary_data: output of summarize_via_tokenbatches containing the summary and score for each batch
        :param Union[str, Path] target_file: the file to save the summary to, defaults to None
        :param bool postprocess: whether to postprocess the summary, defaults to True
        :param str batch_delimiter: text delimiter between summary batches, defaults to "\n\n"
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
        full_summary = batch_delimiter.join(sum_text)
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
                fo.write("\n" * 2 + "---\n\n")
                fo.write(f"Section Scores for {target_file.stem}:\n")
                fo.writelines(scores_text)
                fo.write("\n\n---\n")

        self.logger.info(f"Saved summary to {target_file.resolve()}")

    def summarize_string(
        self,
        input_text: str,
        batch_length: int = None,
        batch_stride: int = None,
        batch_delimiter: str = "\n\n",
        disable_progress_bar: bool = None,
        **kwargs,
    ) -> str:
        """
        summarize_string - generate a summary for a string of text

        :param str input_text: the text to summarize
        :param int batch_length: number of tokens to use in each batch, defaults to None (self.token_batch_length)
        :param int batch_stride: number of tokens to stride between batches, defaults to None (self.batch_stride)
        :param str batch_delimiter: text delimiter between summary batches, defaults to "\n\n"
        :param bool disable_progress_bar: whether to disable the progress bar, defaults to None
        :param kwargs: additional parameters to pass to summarize_via_tokenbatches

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
            disable_progress_bar=disable_progress_bar,
            **kwargs,
        )

        return self.save_summary(
            summary_data=gen_summaries,
            return_string=True,
            batch_delimiter=batch_delimiter,
        )

    def summarize_file(
        self,
        file_path: Union[str, Path],
        output_dir: Union[str, Path] = None,
        lowercase: bool = False,
        batch_length: int = None,
        batch_stride: int = None,
        batch_delimiter: str = "\n\n",
        save_scores: bool = True,
        disable_progress_bar: bool = None,
        **kwargs,
    ) -> Path:
        """
        summarize_file - generate a summary for a text file

        :param Union[str, Path] file_path: The path to the text file.
        :param Union[str, Path] output_dir: The path to the output directory, defaults to None
        :param bool lowercase: whether to lowercase the text, defaults to False
        :param int batch_length: Number of tokens to use in each batch, defaults to None
        :param int batch_stride: Number of tokens to stride between batches, defaults to None
        :param str batch_delimiter: Text delimiter between output summary batches, defaults to "\n\n"
        :param bool save_scores: Whether to save the scores to the output file, defaults to True
        :param bool disable_progress_bar: disable the progress bar, defaults to None
        :return Path: The path to the output file
        """

        file_path = Path(file_path)
        output_dir = Path(output_dir) if output_dir is not None else Path.cwd()

        with open(file_path, "r", encoding="utf-8") as f:
            text = clean(f.read(), lower=lowercase)

        # Generate summaries using token batches
        gen_summaries = self.summarize_via_tokenbatches(
            text,
            batch_length=batch_length,
            batch_stride=batch_stride,
            disable_progress_bar=disable_progress_bar,
            **kwargs,
        )

        # Save the generated summaries to the output file
        output_file = output_dir / f"{file_path.stem}_summary.txt"
        self.save_summary(
            gen_summaries,
            output_file,
            batch_delimiter=batch_delimiter,
            save_scores=save_scores,
        )

        return output_file

    def save_params(
        self,
        output_path: Union[str, Path] = None,
        hf_tag: str = None,
        verbose: bool = False,
    ) -> None:
        """
        save_params - save the parameters of the run to a json file

        :param dict params: parameters to save
        :param Union[str, Path] output_path: directory or filepath to save the parameters to
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
        metadata = {
            "META_huggingface_model": (
                self.model_name_or_path if hf_tag is None else hf_tag
            ),
            "META_date": get_timestamp(),
            "META_textsum_version": textsum.__version__,
        }
        exported_params["METADATA"] = metadata

        self.logger.info(f"Saving parameters to {metadata_path}")
        with open(metadata_path, "w") as write_file:
            json.dump(exported_params, write_file, indent=2)

        logging.debug(f"Saved parameters to {metadata_path}")
        if verbose:
            self.logger.info(f"parameters: {exported_params}")
            print(f"saved parameters to {metadata_path}")

    def __call__(self, input_data, **kwargs):
        """
        Smart __call__ function to decide where to route the inputs based on whether a valid filepath is passed.

        :param input_data: Can be either a string (text to summarize) or a file path.
        :param kwargs: Additional keyword arguments to pass to the summarization methods.
        :return: The summary of the input text, or saves the summary to a file if a file path is provided.

        Example usage:
            summarizer = Summarizer()
            summary = summarizer("This is a test string to summarize.")
            # or
            summary = summarizer("/path/to/textfile.txt")
        """
        MAX_FILEPATH_LENGTH = 300  # est
        if (
            len(str(input_data)) < MAX_FILEPATH_LENGTH
            and isinstance(input_data, (str, Path))
            and Path(input_data).is_file()
        ):
            self.logger.debug("Summarizing from file...")
            return self.summarize_file(file_path=input_data, **kwargs)
        elif isinstance(input_data, str):
            self.logger.debug("Summarizing from string...")
            return self.summarize_string(input_text=input_data, **kwargs)
        else:
            raise ValueError("Input must be a valid string or a file path.")
