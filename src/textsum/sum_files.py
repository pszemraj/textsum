import argparse
import logging
import pprint as pp
import random
from pathlib import Path

import torch
from cleantext import clean
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textsum.summarize import load_model_and_tokenizer, summarize_via_tokenbatches
from textsum.utils import get_mem_footprint, setup_logging, postprocess_booksummary


def summarize_text_file(
    file_path,
    model,
    tokenizer,
    batch_length: int = 2048,
    batch_stride: int = 16,
    lowercase: bool = True,
    **kwargs,
):
    """
    summarize_text_file - given a file path, return a summary of the file

    Args:
        file_path (str): the path to the file to summarize
        model (): the model to use for summarization
        tokenizer (): the tokenizer to use for summarization
        kw

    Returns:
        dict: a dictionary containing the summary and other information
    """
    file_path = Path(file_path)

    ALLOWED_EXTENSIONS = [".txt", ".md", ".rst", ".py", ".ipynb"]
    assert (
        file_path.exists() and file_path.suffix in ALLOWED_EXTENSIONS
    ), f"File {file_path} does not exist or is not a text file"

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = clean(f.read(), lower=lowercase, no_line_breaks=True)

    summary_data = summarize_via_tokenbatches(
        input_text=text,
        model=model,
        tokenizer=tokenizer,
        batch_length=batch_length,
        batch_stride=batch_stride,
        **kwargs,
    )

    return summary_data


def process_summarization(
    summary_data,
    file_path,
    custom_phrases: list = None,
):
    sum_text = [postprocess_booksummary(s["summary"][0]) for s in summary_data]
    sum_scores = [f"\n - {round(s['summary_score'],4)}" for s in summary_data]
    scores_text = "\n".join(sum_scores)
    full_summary = "\n\t".join(sum_text)

    with open(
        file_path,
        "w",
    ) as fo:

        fo.writelines(full_summary)
    with open(
        file_path,
        "a",
    ) as fo:

        fo.write("\n" * 3)
        fo.write(f"\n\nSection Scores for {f.name}:\n")
        fo.writelines(scores_text)
        fo.write("\n\n---\n")


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
        "-lr",
        "-length_ratio",
        "--max_length_ratio",
        target="max_length_ratio",
        type=int,
        default=0.25,
        help="the maximum length of the summary as a ratio of the batch length",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=8,
        help="the minimum length of the summary",
    )
    parser.add_argument(
        "-enc_ngram",
        "--encoder_no_repeat_ngram_size",
        type=int,
        default=3,
        target="encoder_no_repeat_ngram_size",
        help="the encoder no repeat ngram size (from source)",
    )
    parser.add_argument(
        "--no_early_stopping",
        action="store_false",
        target="early_stopping",
        help="do not use early stopping when generating summaries with beam search",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle the input files before summarizing",
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


def main(args):

    logging.info(f"args: {pp.pformat(args)}")
    setup_logging(args.loglevel, args.logfile)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    logging.info(f"using device: {device}")
    # load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    logging.info(f"model size: {get_mem_footprint(model)}")
    # move the model to the device
    model.to(device)

    params = {
        "min_length": args.min_length,
        "max_length": int(args.max_length_ratio * args.batch_length),
        "encoder_no_repeat_ngram_size": args.encoder_no_repeat_ngram_size,
        "repetition_penalty": 2.5,
        "num_beams": args.num_beams,
        "num_beam_groups": 1,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
        "do_sample": False,
    }
    # get the input files
    input_files = list(Path(args.input_dir).glob("*.txt"))

    if args.shuffle:
        logging.info("shuffling input files")
        random.SystemRandom().shuffle(input_files)


def run():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run()
