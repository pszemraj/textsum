"""
    utils.py - Utility functions for the project.
"""

import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)


def get_timestamp() -> str:
    """
    get_timestamp - get a timestamp for the current time
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def regex_gpu_name(input_text: str):
    """backup if not a100"""

    pattern = re.compile(r"(\s([A-Za-z0-9]+\s)+)(\s([A-Za-z0-9]+\s)+)", re.IGNORECASE)
    return pattern.search(input_text).group()


def check_GPU(verbose=False):
    """
    check_GPU - a function in Python that uses the subprocess module and regex to call the `nvidia-smi` command and check the available GPU. the function returns a boolean as to whether the GPU is an A100 or not

    :param verbose: if true, print out which GPU was found if it is not an A100
    """
    # call nvidia-smi
    nvidia_smi = subprocess.run(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    # convert to string
    nvidia_smi = nvidia_smi.stdout.decode("utf-8")
    search_past = "==============================="
    # use regex to find the GPU name. search in the first newline underneath <search_past>
    output_lines = nvidia_smi.split("\n")
    for i, line in enumerate(output_lines):
        if search_past in line:
            break
    # get the next line
    next_line = output_lines[i + 1]
    if verbose:
        print(next_line)
    # use regex to find the GPU name
    try:
        gpu_name = re.search(r"\w+-\w+-\w+", next_line).group()
    except AttributeError:
        logging.debug("Could not find GPU name with initial regex")
        gpu_name = None

    if gpu_name is None:
        # try alternates
        try:
            gpu_name = regex_gpu_name(next_line)
        except Exception as e:
            logging.error(f"Could not find GPU name: {e}")
            return False

    if verbose:
        print(f"GPU found: {gpu_name}")
    # check if it is an A100
    return bool("A100" in gpu_name)


def cstr(s, color="black"):
    """styles a string with a color"""
    return "<text style=color:{}>{}</text>".format(color, s)


def color_print(text: str, c_id="pink"):
    """helper function to print colored text to the terminal"""

    colormap = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "pink": "\033[95m",
        "teal": "\033[96m",
        "grey": "\033[97m",
    }

    print(f"{colormap[c_id]}{text}")


def get_mem_footprint(test_model):
    """
    get_mem_footprint - a helper function for the gradio module to get the memory footprint of a model (for huggingface models)
    """
    fp = test_model.get_memory_footprint() * (10**-9)
    print(f"memory footprint is approx {round(fp, 2)} GB")


def truncate_word_count(text, max_words=512):
    """
    truncate_word_count - a helper function for the gradio module to truncate the text to a max number of words

    :param str text: the text to truncate
    :param int max_words: the max number of words to truncate to (default 512)
    :return dict: a dictionary with the truncated text and a boolean indicating whether the text was truncated
    """

    words = re.split(r"\s+", text)
    processed = {}
    if len(words) > max_words:
        processed["was_truncated"] = True
        processed["truncated_text"] = " ".join(words[:max_words])
    else:
        processed["was_truncated"] = False
        processed["truncated_text"] = text
    return processed


def setup_logging(loglevel, logfile=None) -> None:
    """Setup basic logging
        you will need something like this in your main script:
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
                "--very-verbose",
                dest="loglevel",
                help="set loglevel to DEBUG",
                action="store_const",
                const=logging.DEBUG,
            )
    Args:
        loglevel (int): minimum loglevel for emitting messages
        logfile (str): path to logfile. If None, log to stderr.
    """
    # remove any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    if logfile is None:
        logging.basicConfig(
            level=loglevel,
            stream=sys.stdout,
            format=logformat,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        loglevel = (
            logging.INFO if not loglevel in [logging.DEBUG, logging.INFO] else loglevel
        )
        logging.basicConfig(
            level=loglevel,
            filename=logfile,
            filemode="w",
            format=logformat,
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def postprocess_booksummary(text: str, custom_phrases: list = None) -> str:
    """
    postprocess_booksummary - postprocess the book summary

    :param str text: the text to postprocess
    :param list custom_phrases: custom phrases to remove from the text, defaults to None
    :return str: the postprocessed text
    """
    REMOVAL_PHRASES = [
        "In this section, ",
        "In this lecture, ",
        "In this chapter, ",
        "In this paper, ",
    ]  # the default phrases to remove (from booksum dataset)

    if custom_phrases is not None:
        REMOVAL_PHRASES.extend(custom_phrases)
    for pr in REMOVAL_PHRASES:

        text = text.replace(pr, "")
    return text


def check_bitsandbytes_available():
    """
    check_bitsandbytes_available - check if the bitsandbytes library is available
    """
    try:
        import bitsandbytes
    except ImportError:
        return False
    return True


def enable_tf32():
    """
    enable_tf32 - enables computation in tf32 precision. (requires ampere series GPU or newer)

        See https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/ for details
    """
    logging.debug("Enabling TF32 computation")
    torch.backends.cuda.matmul.allow_tf32 = True
