"""
    utils.py - Utility functions for the project.
"""

import re
from pathlib import Path
from datetime import datetime
from natsort import natsorted
import subprocess


def get_timestamp() -> str:
    """
    get_timestamp - get a timestamp for the current time
    Returns:
        str, the timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def load_examples(src, filetypes=[".txt", ".pdf"]):
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


def load_example_filenames(example_path: str or Path):
    """
    load_example_filenames - a helper function for the gradio module to load examples
    Returns:
        dict, the examples (filename:full path)
    """
    example_path = Path(example_path)
    # load the examples into a list
    examples = {f.name: f for f in example_path.glob("*.txt")}
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
