"""
utils.py - Utility functions for the project.
"""

import logging
import re
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


def validate_pytorch2(torch_version: str = None):
    torch_version = torch.__version__ if torch_version is None else torch_version

    pattern = r"^2\.\d+(\.\d+)*"

    return True if re.match(pattern, torch_version) else False


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

    log_format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    debug_format = (
        "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d - %(message)s"
    )
    if logfile is None:
        logging.basicConfig(
            level=loglevel,
            stream=sys.stdout,
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logfile = Path(logfile)
        loglevel = (
            logging.INFO
            if loglevel not in [logging.DEBUG, logging.INFO, logging.WARNING]
            else loglevel
        )
        if loglevel == logging.DEBUG:
            logfile.unlink(missing_ok=True)

        logging.basicConfig(
            level=loglevel,
            filename=logfile,
            filemode="w",
            format=debug_format if loglevel == logging.DEBUG else log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def postprocess_booksummary(text: str, custom_phrases: list = None) -> str:
    """
    Postprocess the book summary by removing specified introductory phrases if they
    appear at the beginning of the text (case-insensitive).

    :param str text: The text to postprocess.
    :param list custom_phrases: Custom phrases to remove from the text, defaults to None.
    :return str: The postprocessed text.
    """
    REMOVAL_PHRASES = [
        "In this section, ",
        "In this lecture, ",
        "In this chapter, ",
        "In this paper, ",
    ]

    if custom_phrases:
        REMOVAL_PHRASES.extend(custom_phrases)

    for phrase in REMOVAL_PHRASES:
        if text.lower().startswith(phrase.lower()):
            text = text[len(phrase) :]
            break  # Stop after the first match to preserve other phrases

    return text.strip()


def check_bitsandbytes_available():
    """
    check_bitsandbytes_available - check if the bitsandbytes library is available
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False
    return True


def check_ampere_gpu() -> None:
    """
    Check if the GPU supports NVIDIA Ampere or later and enable TF32 in PyTorch if it does.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.info("No GPU detected, running on CPU.")
        return

    try:
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        major, minor = capability

        # Check if Ampere or newer (compute capability >= 8.0)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) supports NVIDIA Ampere or later, enabled TF32 in PyTorch."
            )
        else:
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"{gpu_name} (compute capability {major}.{minor}) is not NVIDIA Ampere or later."
            )

    except Exception as e:
        logging.warning(f"Error occurred while checking GPU: {e}")
