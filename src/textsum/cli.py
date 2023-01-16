"""
cli.py - a module containing functions for the command line interface (to run the summarization on a directory of files)
        #TODO: add a function to summarize a single file

usage: textsum-dir [-h] [-o OUTPUT_DIR] [-m MODEL_NAME] [-batch BATCH_LENGTH] [-stride BATCH_STRIDE] [-nb NUM_BEAMS]
                   [-l2 LENGTH_PENALTY] [-r2 REPETITION_PENALTY] [--no_cuda] [-length_ratio MAX_LENGTH_RATIO] [-ml MIN_LENGTH]
                   [-enc_ngram ENCODER_NO_REPEAT_NGRAM_SIZE] [-dec_ngram NO_REPEAT_NGRAM_SIZE] [--no_early_stopping] [--shuffle]
                   [--lowercase] [-v] [-vv] [-lf LOGFILE]
                   input_dir

Summarize text files in a directory

positional arguments:
  input_dir             the directory containing the input files

"""
import argparse
import logging
import pprint as pp
import random
import sys
import warnings
from pathlib import Path

import torch
from cleantext import clean
from tqdm.auto import tqdm

from textsum.summarize import (
    load_model_and_tokenizer,
    save_params,
    summarize_via_tokenbatches,
)
from textsum.utils import get_mem_footprint, postprocess_booksummary, setup_logging


def summarize_text_file(
    file_path: str or Path,
    model,
    tokenizer,
    batch_length: int = 4096,
    batch_stride: int = 16,
    lowercase: bool = False,
    **kwargs,
) -> dict:
    """
    summarize_text_file - given a file path, summarize the text in the file

    :param str or Path file_path: the path to the file to summarize
    :param model: the model to use for summarization
    :param tokenizer: the tokenizer to use for summarization
    :param int batch_length: length of each batch in tokens to summarize, defaults to 4096
    :param int batch_stride: stride between batches in tokens, defaults to 16
    :param bool lowercase: whether to lowercase the text before summarizing, defaults to False
    :return: a dictionary containing the summary and other information
    """
    file_path = Path(file_path)
    ALLOWED_EXTENSIONS = [".txt", ".md", ".rst", ".py", ".ipynb"]
    assert (
        file_path.exists() and file_path.suffix in ALLOWED_EXTENSIONS
    ), f"File {file_path} does not exist or is not a text file"

    logging.info(f"Summarizing {file_path}")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = clean(f.read(), lower=lowercase, no_line_breaks=True)
    logging.debug(
        f"Text length: {len(text)}. batch length: {batch_length} batch stride: {batch_stride}"
    )
    summary_data = summarize_via_tokenbatches(
        input_text=text,
        model=model,
        tokenizer=tokenizer,
        batch_length=batch_length,
        batch_stride=batch_stride,
        **kwargs,
    )
    logging.info(f"Finished summarizing {file_path}")
    return summary_data


def process_summarization(
    summary_data: dict,
    target_file: str or Path,
    custom_phrases: list = None,
    save_scores: bool = True,
) -> None:
    """
    process_summarization - given a dictionary of summary data, save the summary to a file

    :param dict summary_data: a dictionary containing the summary and other information (output from summarize_text_file)
    :param str or Path target_file: the path to the file to save the summary to
    :param list custom_phrases: a list of custom phrases to remove from each summary (relevant for dataset specific repeated phrases)
    :param bool save_scores: whether to write the scores to a file
    """
    target_file = Path(target_file).resolve()
    if target_file.exists():
        warnings.warn(f"File {target_file} exists, overwriting")

    sum_text = [
        postprocess_booksummary(
            s["summary"][0],
            custom_phrases=custom_phrases,
        )
        for s in summary_data
    ]
    sum_scores = [f"\n - {round(s['summary_score'],4)}" for s in summary_data]
    scores_text = "\n".join(sum_scores)
    full_summary = "\n\t".join(sum_text)

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


def get_parser():
    """
    get_parser - a function that returns an argument parser for the sum_files script

    :return argparse.ArgumentParser: the argument parser
    """
    parser = argparse.ArgumentParser(
        description="Summarize text files in a directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        dest="output_dir",
        help="directory to write the output files (if None, writes to input_dir/summarized)",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="pszemraj/long-t5-tglobal-base-16384-book-summary",
        help="the name of the model to use for summarization",
    )
    parser.add_argument(
        "-batch",
        "--batch_length",
        dest="batch_length",
        type=int,
        default=4096,
        help="the length of each batch",
    )
    parser.add_argument(
        "-stride",
        "--batch_stride",
        type=int,
        default=16,
        help="the stride of each batch",
    )
    parser.add_argument(
        "-nb",
        "--num_beams",
        type=int,
        default=4,
        help="the number of beams to use for beam search",
    )
    parser.add_argument(
        "-l2",
        "--length_penalty",
        type=float,
        default=0.8,
        help="the length penalty to use for decoding",
    )
    parser.add_argument(
        "-r2",
        "--repetition_penalty",
        type=float,
        default=2.5,
        help="the repetition penalty to use for beam search",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="flag to not use cuda if available",
    )
    parser.add_argument(
        "-length_ratio",
        "--max_length_ratio",
        dest="max_length_ratio",
        type=int,
        default=0.25,
        help="the maximum length of the summary as a ratio of the batch length",
    )
    parser.add_argument(
        "-ml",
        "--min_length",
        type=int,
        default=8,
        help="the minimum length of the summary",
    )
    parser.add_argument(
        "-enc_ngram",
        "--encoder_no_repeat_ngram_size",
        type=int,
        default=4,
        dest="encoder_no_repeat_ngram_size",
        help="encoder no repeat ngram size (input text). smaller values mean more unique summaries",
    )
    parser.add_argument(
        "-dec_ngram",
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        dest="no_repeat_ngram_size",
        help="the decoder no repeat ngram size (output text)",
    )
    parser.add_argument(
        "--no_early_stopping",
        action="store_false",
        dest="early_stopping",
        help="whether to use early stopping. this disables the early_stopping value",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle the input files before summarizing",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="whether to lowercase the input text",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very_verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    parser.add_argument(
        "-lf",
        "--log_file",
        dest="logfile",
        type=str,
        default=None,
        help="path to the log file. this will set loglevel to INFO (if not set) and write to the file",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="the directory containing the input files",
    )

    # if there are no args, print the help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser


def main(args):
    """
    main - the main function for the script

    :param argparse.Namespace args: the arguments for the script
    """
    setup_logging(args.loglevel, args.logfile)
    logging.info("starting summarization")
    logging.info(f"args: {pp.pformat(args)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    logging.info(f"using device: {device}")
    # load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, use_cuda=not args.no_cuda
    )

    logging.info(f"model size: {get_mem_footprint(model)}")
    # move the model to the device
    model.to(device)

    params = {
        "min_length": args.min_length,
        "max_length": int(args.max_length_ratio * args.batch_length),
        "encoder_no_repeat_ngram_size": args.encoder_no_repeat_ngram_size,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "num_beams": args.num_beams,
        "num_beam_groups": 1,
        "length_penalty": args.length_penalty,
        "early_stopping": args.early_stopping,
        "do_sample": False,
    }
    # get the input files
    input_files = list(Path(args.input_dir).glob("*.txt"))

    if args.shuffle:
        logging.info("shuffling input files")
        random.SystemRandom().shuffle(input_files)

    # get the output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.input_dir) / "summarized"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    # get the batches
    for f in tqdm(input_files):

        outpath = output_dir / f"{f.stem}.summary.txt"
        summary_data = summarize_text_file(
            file_path=f,
            model=model,
            tokenizer=tokenizer,
            batch_length=args.batch_length,
            batch_stride=args.batch_stride,
            lowercase=args.lowercase,
            **params,
        )
        process_summarization(
            summary_data=summary_data, target_file=outpath, save_scores=True
        )

    logging.info(f"finished summarization loop - output dir: {output_dir.resolve()}")
    save_params(params=params, output_dir=output_dir, hf_tag=args.model_name)

    logging.info("finished summarizing files")


def run():
    """
    run - main entry point for the script
    """

    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run()
