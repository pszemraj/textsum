"""
    utils.py - Utility functions for the project.
"""

import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)

from natsort import natsorted

# ------------------------- #

TEXT_EXAMPLE_URLS = {
    "whisper_lecture": "https://pastebin.com/raw/X9PEgS2w",
    "hf_blog_clip": "https://pastebin.com/raw/1RMg1Naz",
}

# ------------------------- #


def get_timestamp() -> str:
    """
    get_timestamp - get a timestamp for the current time
    Returns:
        str, the timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


import re
import subprocess


def regex_gpu_name(input_text):
    """backup if not a100"""

    pattern = re.compile(r"(\s([A-Za-z0-9]+\s)+)(\s([A-Za-z0-9]+\s)+)", re.IGNORECASE)
    return pattern.search(input_text).group()


def check_GPU(verbose=False):
    """
    check_GPU - a function in Python that uses the subprocess module and regex to call the `nvidia-smi` command and check the available GPU. the function returns a boolean as to whether the GPU is an A100 or not

    :param verbose: if true, print out which GPU was found if it is not an A100
    """
    # call nvidia-smi
    nvidia_smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE)
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
        gpu_name = None
    if gpu_name is None:
        # try alternates
        try:
            gpu_name = regex_gpu_name(next_line)
        except Exception as e:
            print(f"unable to lookup GPU from output:\t{e}")
            return False

    if verbose:
        print(f"GPU found: {gpu_name}")
    # check if it is an A100
    if "A100" in gpu_name:
        return True
    else:
        return False


def truncate_word_count(text, max_words=512):
    """
    truncate_word_count - a helper function for the gradio module
    Parameters
    ----------
    text : str, required, the text to be processed
    max_words : int, optional, the maximum number of words, default=512
    Returns
    -------
    dict, the text and whether it was truncated
    """
    # split on whitespace with regex
    words = re.split(r"\s+", text)
    processed = {}
    if len(words) > max_words:
        processed["was_truncated"] = True
        processed["truncated_text"] = " ".join(words[:max_words])
    else:
        processed["was_truncated"] = False
        processed["truncated_text"] = text
    return processed


def load_pdf_examples(src, filetypes=[".txt", ".pdf"]):
    """
    load_examples - a helper function for the gradio module to load examples
    Returns:
        list of str, the examples
    """
    src = Path(src)
    src.mkdir(exist_ok=True)

    pdf_url = (
        "https://www.dropbox.com/s/y92xy7o5qb88yij/all_you_need_is_attention.pdf?dl=1"
    )
    subprocess.run(["wget", pdf_url, "-O", src / "all_you_need_is_attention.pdf"])
    examples = [f for f in src.iterdir() if f.suffix in filetypes]
    examples = natsorted(examples)
    # load the examples into a list
    text_examples = []
    for example in examples:
        with open(example, "r") as f:
            text = f.read()
            text_examples.append([text, "base", 2, 1024, 0.7, 3.5, 3])

    return text_examples


def load_text_examples(
    urls: dict = TEXT_EXAMPLE_URLS, target_dir: str or Path = None
) -> Path:
    """
    load_text_examples - load the text examples from the web to a directory

    :param dict urls: the urls to the text examples, defaults to TEXT_EXAMPLE_URLS
    :param strorPath target_dir: the path to the target directory, defaults to the current working directory
    :return Path: the path to the directory containing the text examples
    """
    target_dir = Path.cwd() if target_dir is None else Path(target_dir)
    target_dir.mkdir(exist_ok=True)

    for name, url in urls.items():  # download the examples
        subprocess.run(["wget", url, "-O", target_dir / f"{name}.txt"])

    return target_dir


def load_example_filenames(example_path: str or Path, ext: list = [".txt", ".md"]):
    """
    load_example_filenames - load the example filenames from a directory

    :param strorPath example_path: the path to the examples directory
    :param list ext: the file extensions to load (default: [".txt", ".md"])
    :return dict: the example filenames
    """
    example_path = Path(example_path)
    if not example_path.exists():
        # download the examples
        logging.info("Downloading the examples...")
        example_path = load_text_examples(target_dir=example_path)

    # load the examples into a list
    examples = {f.name: f.resolve() for f in example_path.glob("*") if f.suffix in ext}
    logging.info(f"Loaded {len(examples)} examples from {example_path}")
    return examples


def saves_summary(summarize_output, outpath: str or Path = None, add_signature=True):
    """

    saves_summary - save the summary generated from summarize_via_tokenbatches() to a text file

            _summaries = summarize_via_tokenbatches(
              text,
              batch_length=token_batch_length,
              batch_stride=batch_stride,
              **settings,
          )
    """

    outpath = (
        Path.cwd() / f"document_summary_{get_timestamp()}.txt"
        if outpath is None
        else Path(outpath)
    )
    sum_text = [s["summary"][0] for s in summarize_output]
    sum_scores = [f"\n - {round(s['summary_score'],4)}" for s in summarize_output]
    scores_text = "\n".join(sum_scores)
    full_summary = "\n\t".join(sum_text)

    with open(
        outpath,
        "w",
    ) as fo:
        if add_signature:
            fo.write(
                "Generated with the Document Summarization space :) https://hf.co/spaces/pszemraj/document-summarization\n\n"
            )
        fo.writelines(full_summary)
    with open(
        outpath,
        "a",
    ) as fo:

        fo.write("\n" * 3)
        fo.write(f"\n\nSection Scores:\n")
        fo.writelines(scores_text)
        fo.write("\n\n---\n")

    return outpath


def setup_logging(loglevel, logfile=None):
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


def postprocess_booksummary(text: str, custom_phrases: list = None):
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
