"""
cli.py - Command line interface for textsum.

Usage:
    textsum-dir --help
"""
import logging
import pprint as pp
import random
from pathlib import Path
from typing import Optional

import fire
from tqdm.auto import tqdm

import textsum
from textsum.summarize import Summarizer
from textsum.utils import enable_tf32, setup_logging


def main(
    input_dir: str,
    output_dir: Optional[str] = None,
    model: str = "pszemraj/long-t5-tglobal-base-16384-book-summary",
    no_cuda: bool = False,
    tf32: bool = False,
    force_cache: bool = False,
    load_in_8bit: bool = False,
    compile: bool = False,
    optimum_onnx: bool = False,
    batch_length: int = 4096,
    batch_stride: int = 16,
    num_beams: int = 4,
    length_penalty: float = 0.8,
    repetition_penalty: float = 2.5,
    max_length_ratio: float = 0.25,
    min_length: int = 8,
    encoder_no_repeat_ngram_size: int = 4,
    no_repeat_ngram_size: int = 3,
    early_stopping: bool = True,
    shuffle: bool = False,
    lowercase: bool = False,
    loglevel: Optional[int] = 30,
    logfile: Optional[str] = None,
    file_extension: str = "txt",
    skip_completed: bool = False,
):
    """
    Main function to summarize text files in a directory.

    Args:
        input_dir (str, required): The directory containing the input files.
        output_dir (str, optional): Directory to write the output files. If None, writes to input_dir/summarized.
        model (str, optional): The name of the model to use for summarization. Default: "pszemraj/long-t5-tglobal-base-16384-book-summary".
        no_cuda (bool, optional): Flag to not use cuda if available. Default: False.
        tf32 (bool, optional): Enable tf32 data type for computation (requires ampere series GPU or newer). Default: False.
        force_cache (bool, optional): Force the use_cache flag to True in the Summarizer. Default: False.
        load_in_8bit (bool, optional): Flag to load the model in 8 bit precision (requires bitsandbytes). Default: False.
        compile (bool, optional): Compile the model for inference (requires torch 2.0+). Default: False.
        optimum_onnx (bool, optional): Optimize the model for inference (requires onnxruntime-tools). Default: False.
        batch_length (int, optional): The length of each batch. Default: 4096.
        batch_stride (int, optional): The stride of each batch. Default: 16.
        num_beams (int, optional): The number of beams to use for beam search. Default: 4.
        length_penalty (float, optional): The length penalty to use for decoding. Default: 0.8.
        repetition_penalty (float, optional): The repetition penalty to use for beam search. Default: 2.5.
        max_length_ratio (float, optional): The maximum length of the summary as a ratio of the batch length. Default: 0.25.
        min_length (int, optional): The minimum length of the summary. Default: 8.
        encoder_no_repeat_ngram_size (int, optional): Encoder no repeat ngram size (input text). Smaller values mean more unique summaries. Default: 4.
        no_repeat_ngram_size (int, optional): The decoder no repeat ngram size (output text). Default: 3.
        early_stopping (bool, optional): Whether to use early stopping. Default: True.
        shuffle (bool, optional): Shuffle the input files before summarizing. Default: False.
        lowercase (bool, optional): Whether to lowercase the input text. Default: False.
        loglevel (int, optional): The log level to use (default: 20 - INFO). Default: 30.
        logfile (str, optional): Path to the log file. This will set loglevel to INFO (if not set) and write to the file.
        file_extension (str, optional): The file extension to use when searching for input files.,  defaults to "txt"
        skip_completed (bool, optional): Skip files that have already been summarized. Default: False.

    Returns:
        None
    """
    setup_logging(loglevel, logfile)
    logging.info("starting textsum cli")
    logging.info(f"textsum version:\t{textsum.__version__}")

    params = {
        "min_length": min_length,
        "encoder_no_repeat_ngram_size": encoder_no_repeat_ngram_size,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
        "num_beams": num_beams,
        "num_beam_groups": 1,
        "length_penalty": length_penalty,
        "early_stopping": early_stopping,
        "do_sample": False,
    }

    if tf32:
        enable_tf32()  # enable tf32 for computation

    summarizer = Summarizer(
        model_name_or_path=model,
        use_cuda=not no_cuda,
        token_batch_length=batch_length,
        batch_stride=batch_stride,
        max_length_ratio=max_length_ratio,
        load_in_8bit=load_in_8bit,
        compile_model=compile,
        optimum_onnx=optimum_onnx,
        force_cache=force_cache,
        **params,
    )
    summarizer.print_config()
    logging.info(summarizer.config)
    # get the input files
    input_files = list(Path(input_dir).glob(f"*.{file_extension}"))
    logging.info(f"found {len(input_files)} input files")

    if shuffle:
        logging.info("shuffling input files")
        random.SystemRandom().shuffle(input_files)

    # get the output directory
    output_dir = Path(output_dir) if output_dir else Path(input_dir) / "summarized"
    output_dir.mkdir(exist_ok=True, parents=True)

    failed_files = []
    completed_files = []
    for f in tqdm(input_files, desc="summarizing files"):
        _prospective_output_file = output_dir / f"{f.stem}_summary.txt"
        if skip_completed and _prospective_output_file.exists():
            logging.info(f"skipping file (found existing summary):\t{str(f)}")
            continue
        try:
            _ = summarizer.summarize_file(
                file_path=f, output_dir=output_dir, lowercase=lowercase
            )
            completed_files.append(str(f))
        except Exception as e:
            logging.error(f"failed to summarize file:\t{f}")
            logging.error(e)
            print(e)
            failed_files.append(f)
            if isinstance(e, RuntimeError):
                # if a runtime error occurs, exit immediately
                logging.error("Not continuing summarization due to runtime error")
                failed_files.extend(input_files[input_files.index(f) + 1 :])
                break

    logging.info(f"failed to summarize {len(failed_files)} files")
    if len(failed_files) > 0:
        logging.info(f"failed files:\n\t{pp.pformat(failed_files)}")

    logging.debug("saving summarizer params and config")
    summarizer.save_params(output_path=output_dir, hf_tag=model)
    summarizer.save_config(output_dir / "textsum_config.json")
    logging.info(
        f"finished summarizing files - output dir:\n\t{str(output_dir.resolve())}"
    )


def run():
    """Entry point for console_scripts"""
    fire.Fire(main)


if __name__ == "__main__":
    run()
