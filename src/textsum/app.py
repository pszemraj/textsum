"""
app.py - a module to run the text summarization app (gradio interface)
"""
import contextlib
import logging
import os
import random
import re
import time
from pathlib import Path

os.environ["USE_TORCH"] = "1"
os.environ["DEMO_MAX_INPUT_WORDS"] = "2048"  # number of words to truncate input to
os.environ["DEMO_MAX_INPUT_PAGES"] = "20"  # number of pages to truncate PDFs to
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # parallelism is buggy with gradio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import gradio as gr
import nltk
from cleantext import clean
from doctr.models import ocr_predictor

from textsum.pdf2text import convert_PDF_to_Text
from textsum.summarize import Summarizer
from textsum.utils import truncate_word_count, get_timestamp

_here = Path.cwd()

nltk.download("stopwords")  # TODO=find where this requirement originates from


def proc_submission(
    input_text: str,
    num_beams: int,
    token_batch_length: int,
    length_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_input_words: int = 1024,
):
    """
    proc_submission - a helper function for the gradio module to process submissions

    Args:
        input_text (str): the input text to summarize
        num_beams (int): the number of beams to use
        token_batch_length (int): the length of the token batches to use
        length_penalty (float): the length penalty to use
        repetition_penalty (float): the repetition penalty to use
        no_repeat_ngram_size (int): the no repeat ngram size to use
        max_input_length (int, optional): the maximum input length to use. Defaults to 768.

    Returns:
        str in HTML format, string of the summary, str of score
    """

    global summarizer
    max_input_words = (
        int(os.environ["DEMO_MAX_INPUT_WORDS"])
        if int(os.environ["DEMO_MAX_INPUT_WORDS"]) > 0
        else max_input_words
    )
    settings = {
        "length_penalty": float(length_penalty),
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "encoder_no_repeat_ngram_size": 4,
        "num_beams": int(num_beams),
        "min_length": 4,
        "max_length": int(token_batch_length // 4),
        "early_stopping": True,
        "do_sample": False,
    }

    if "summarizer" not in globals():
        logging.info("model not loaded, reloading now")
        summarizer = Summarizer(
            use_cuda=True,
            token_batch_length=token_batch_length,
            **settings,
        )

    st = time.perf_counter()
    history = {}
    clean_text = clean(input_text, lower=False)
    processed = truncate_word_count(
        clean_text,
        max_words=max_input_words,
    )

    if processed["was_truncated"]:
        tr_in = processed["truncated_text"]
        input_wc = re.split(r"\s+", input_text)

        msg = f"""
        <div style="background-color: #FFA500; color: white; padding: 20px;">
        <h3>Warning</h3>
        <p>Input text was truncated to {max_input_words} words. That's about {100*max_input_words/len(input_wc):.2f}% of the submission.</p>
        </div>
        """  # create elaborate HTML warning message
        logging.warning(msg)
        history["WARNING"] = msg
    else:
        tr_in = input_text
        msg = None

    if len(input_text) < 50:
        msg = f"""
        <div style="background-color: #880808; color: white; padding: 20px;">
        <h3>Warning</h3>
        <p>Input text is too short to summarize. Detected {len(input_text)} characters.
        Please load text by selecting an example from the dropdown menu or by pasting text into the text box.</p>
        </div>
        """  # no-input warning
        logging.warning(msg)
        logging.warning("RETURNING EMPTY STRING")
        history["WARNING"] = msg

        return msg, "", []

    processed_outputs = summarizer.summarize_via_tokenbatches(
        input_text=tr_in,
        batch_length=token_batch_length,
    )  # get the summaries

    # reformat output
    history["Summary Scores"] = "<br><br>"
    sum_text = [
        f"\tSection {i}: " + s["summary"][0] for i, s in enumerate(processed_outputs)
    ]
    sum_scores = [
        f"\tSection {i}: {round(s['summary_score'],4)}"
        for i, s in enumerate(processed_outputs)
    ]

    sum_text_out = "\n".join(sum_text)
    scores_out = "\n".join(sum_scores)
    rt = round((time.perf_counter() - st) / 60, 2)
    logging.info(f"Runtime: {rt} minutes")
    html = ""
    html += f"<p>Runtime: {rt} minutes on CPU</p>"
    if msg is not None:
        html += msg

    html += ""

    summary_file = _here / f"summarized_{get_timestamp()}.txt"
    summarizer.save_summary(
        summary_data=processed_outputs,
        target_file=summary_file,
    )

    return html, sum_text_out, scores_out, summary_file


def load_uploaded_file(file_obj, max_pages=20) -> str:
    """
    load_uploaded_file - loads a file added by the user

    :param file_obj: a file object from gr.File()
    :param int max_pages: the maximum number of pages to convert from a PDF
    :return str: the text from the file
    """

    global ocr_model
    max_pages = (
        int(os.environ["DEMO_MAX_INPUT_PAGES"])
        if int(os.environ["DEMO_MAX_INPUT_PAGES"]) > 0
        else max_pages
    )
    logging.info(f"Loading file, truncating to {max_pages} pages for PDFs")
    if isinstance(file_obj, list):
        file_obj = file_obj[0]
    file_path = Path(file_obj.name)
    try:
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            text = clean(raw_text, lower=False)
        elif file_path.suffix == ".pdf":
            logging.info(f"Loading PDF file {file_path}")
            conversion_stats = convert_PDF_to_Text(
                file_path,
                ocr_model=ocr_model,
                max_pages=max_pages,
            )
            text = conversion_stats["converted_text"]
        else:
            logging.error(f"Unknown file type {file_path.suffix}")
            text = "ERROR - check example path"

        return text
    except Exception as e:
        logging.info(f"Trying to load file with path {file_path}, error: {e}")
        return "Error: Could not read file. Ensure that it is a valid text file with encoding UTF-8 if text, and a PDF if PDF."


def main():
    logging.info(f"Starting app instance. Files will be saved to {str(_here)}")

    summarizer = Summarizer()

    logging.info("Loading OCR model")
    with contextlib.redirect_stdout(None):
        ocr_model = ocr_predictor(
            "db_resnet50",
            "crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=True,
        )  # mostly to pre-download the model

    demo = gr.Blocks()
    with demo:

        gr.Markdown("# Summarization UI with `textsum`")
        gr.Markdown(
            f"""
            This is an example use case for fine-tuned long document transformers.
            - Model: `{summarizer.model_name_or_path}`
            - this demo created with the [textsum](https://github.com/pszemraj/textsum) library + gradio.
            """
        )
        with gr.Column():

            gr.Markdown("## Load Inputs & Select Parameters")
            gr.Markdown(
                "Enter text below in the text area. The text will be summarized [using the selected parameters](https://huggingface.co/blog/how-to-generate). Optionally load an example below or upload a file. (`.txt` or `.pdf` - _[link to guide](https://i.imgur.com/c6Cs9ly.png)_)"
            )
            with gr.Row(variant="compact"):
                with gr.Column(scale=0.5, variant="compact"):

                    num_beams = gr.Radio(
                        choices=[2, 3, 4],
                        label="Beam Search: # of Beams",
                        value=2,
                    )
                with gr.Column(variant="compact"):

                    uploaded_file = gr.File(
                        label="File Upload",
                        file_count="single",
                        type="file",
                    )
            with gr.Row():
                input_text = gr.Textbox(
                    lines=4,
                    label="Input Text (for summarization)",
                    placeholder="Enter text to summarize, the text will be cleaned and truncated on Spaces. Narrative, academic (both papers and lecture transcription), and article text work well. May take a bit to generate depending on the input text :)",
                )
                with gr.Column(min_width=100, scale=0.5):

                    load_file_button = gr.Button("Upload File")

        with gr.Column():
            gr.Markdown("## Generate Summary")
            gr.Markdown(
                "Summarization should take ~1-2 minutes for most settings, but may extend up to 5-10 minutes in some scenarios."
            )
            summarize_button = gr.Button(
                "Summarize!",
                variant="primary",
            )

            output_text = gr.HTML("<p><em>Output will appear below:</em></p>")
            gr.Markdown("### Summary Output")
            summary_text = gr.Textbox(
                label="Summary", placeholder="The generated summary will appear here"
            )
            gr.Markdown(
                "The summary scores can be thought of as representing the quality of the summary. less-negative numbers (closer to 0) are better:"
            )
            summary_scores = gr.Textbox(
                label="Summary Scores", placeholder="Summary scores will appear here"
            )

            text_file = gr.File(
                label="Download Summary as Text File",
                file_count="single",
                type="file",
                interactive=False,
            )

        gr.Markdown("---")
        with gr.Column():
            gr.Markdown("### Advanced Settings")
            with gr.Row(variant="compact"):
                length_penalty = gr.inputs.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    label="length penalty",
                    default=0.7,
                    step=0.05,
                )
                token_batch_length = gr.Radio(
                    choices=[512, 768, 1024, 1536],
                    label="token batch length",
                    value=1024,
                )

            with gr.Row(variant="compact"):
                repetition_penalty = gr.inputs.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    label="repetition penalty",
                    default=3.5,
                    step=0.1,
                )
                no_repeat_ngram_size = gr.Radio(
                    choices=[2, 3, 4],
                    label="no repeat ngram size",
                    value=3,
                )
        with gr.Column():
            gr.Markdown("### About the Model")
            gr.Markdown(
                "Model(s) are fine-tuned on the [BookSum dataset](https://arxiv.org/abs/2105.08209).The goal was to create a model that can generalize well and is useful in summarizing lots of text in academic and daily usage."
            )
            gr.Markdown("---")

        load_file_button.click(
            fn=load_uploaded_file, inputs=uploaded_file, outputs=[input_text]
        )

        summarize_button.click(
            fn=proc_submission,
            inputs=[
                input_text,
                num_beams,
                token_batch_length,
                length_penalty,
                repetition_penalty,
                no_repeat_ngram_size,
            ],
            outputs=[output_text, summary_text, summary_scores, text_file],
        )

    demo.launch(enable_queue=True, share=True)


def run():
    """
    run - main entry point for the app
    """
    main()


if __name__ == "__main__":
    run()
