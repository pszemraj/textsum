import argparse
import logging
import pprint as pp
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from cleantext import clean

from textsum.summarize import load_model_and_tokenizer, summarize_via_tokenbatches
from textsum.utils import setup_logging, get_mem_footprint


def summarize_text_file(file_path, model, tokenizer, **kwargs):
    """
    summarize_text_file - given a file path, return a summary of the file

    Args:
        file_path (str): the path to the file to summarize
        model (): the model to use for summarization
        tokenizer (): the tokenizer to use for summarization

    Returns:
        str: the summary of the file
    """
    with open(file_path, "r") as f:
        text = f.read()

    summary = summarize_via_tokenbatches(text, model, tokenizer, **kwargs)

    return summary


def get_parser():
    """
    get_parser - a function that returns an argument parser for the sum_files script

    :return argparse.ArgumentParser: the argument parser
    """
    parser = argparse.ArgumentParser(
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="the directory containing the input files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        target="output_dir",
        help="directory to write the output files (if None, writes to input_dir/summarized)",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="sshleifer/distilbart-xsum-12-6",
        help="the name of the model to use for summarization",
    )
    parser.add_argument(
        "-bs",
        "--batch_length",
        target="batch_length",
        type=int,
        default=4096,
        help="the length of each batch",
    )
    parser.add_argument(
        "--batch_stride",
        type=int,
        default=16,
        help="the stride of each batch",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="the number of beams to use for beam search",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.8,
        help="the length penalty to use for beam search",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="flag to not use cuda if available",
    )
    parser.add_argument(
        "--max_length_ratio",
        target="max_length_ratio",
        type=int,
        default=140,
        help="the maximum length of the summary",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=55,
        help="the minimum length of the summary",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="flag to use early stopping",
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
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()
    logging.info(f"args: {pp.pformat(args)}")
    setup_logging(args.loglevel, args.logfile)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    logging.info(f"using device: {device}")
    # load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # move the model to the device
    model.to(device)

    # get the input files
    input_files = Path(args.input_dir).glob("*.txt")
