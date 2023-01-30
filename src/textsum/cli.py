"""
cli.py - a module containing functions for the command line interface (to run the summarization on a directory of files)

usage: textsum-dir [-h] [-o OUTPUT_DIR] [-m MODEL_NAME] [--no_cuda] [--tf32] [-8bit]
                   [-batch BATCH_LENGTH] [-stride BATCH_STRIDE] [-nb NUM_BEAMS]
                   [-l2 LENGTH_PENALTY] [-r2 REPETITION_PENALTY]
                   [-length_ratio MAX_LENGTH_RATIO] [-ml MIN_LENGTH]
                   [-enc_ngram ENCODER_NO_REPEAT_NGRAM_SIZE] [-dec_ngram NO_REPEAT_NGRAM_SIZE]
                   [--no_early_stopping] [--shuffle] [--lowercase] [-v] [-vv] [-lf LOGFILE]
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
from pathlib import Path

from tqdm.auto import tqdm

from textsum.summarize import Summarizer
from textsum.utils import enable_tf32, setup_logging


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
        "--no_cuda",
        action="store_true",
        help="flag to not use cuda if available",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        dest="tf32",
        help="enable tf32 data type for computation (requires ampere series GPU or newer)",
    )
    parser.add_argument(
        "-8bit",
        "--load_in_8bit",
        action="store_true",
        dest="load_in_8bit",
        help="flag to load the model in 8 bit precision (requires bitsandbytes)",
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

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)  # no args, print help
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

    params = {
        "min_length": args.min_length,
        "encoder_no_repeat_ngram_size": args.encoder_no_repeat_ngram_size,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "num_beams": args.num_beams,
        "num_beam_groups": 1,
        "length_penalty": args.length_penalty,
        "early_stopping": args.early_stopping,
        "do_sample": False,
    }

    if args.tf32:
        enable_tf32()  # enable tf32 for computation

    summarizer = Summarizer(
        model_name_or_path=args.model_name,
        use_cuda=not args.no_cuda,
        token_batch_length=args.batch_length,
        batch_stride=args.batch_stride,
        max_length_ratio=args.max_length_ratio,
        load_in_8bit=args.load_in_8bit,
        **params,
    )

    # get the input files
    input_files = list(Path(args.input_dir).glob("*.txt"))
    logging.info(f"found {len(input_files)} input files")

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

        _ = summarizer.summarize_file(
            file_path=f, output_dir=output_dir, lowercase=args.lowercase
        )

    logging.info(f"finished summarization loop - output dir: {output_dir.resolve()}")
    summarizer.save_params(output_path=output_dir, hf_tag=args.model_name)
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
