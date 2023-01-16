import json
import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textsum.utils import get_timestamp


def load_model_and_tokenizer(model_name: str, use_cuda: bool = True):
    """
    load_model_and_tokenizer - a function that loads a model and tokenizer from huggingface

    Args:
        model_name (str): the name of the model to load from huggingface
        use_cuda (bool, optional): whether to use cuda. Defaults to True.
    Returns:
        AutoModelForSeq2SeqLM: the model
        AutoTokenizer: the tokenizer
    """
    logger = logging.getLogger(__name__)
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    logger.debug(f"loading model {model_name} to {device}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loaded model {model_name} to {device}")
    return model, tokenizer


def summarize_and_score(
    ids, mask, model, tokenizer, is_general_attention_model=True, **kwargs
):
    """
    summarize_and_score - given a batch of ids and a mask, return a summary and a score for the summary

    Args:
        ids (): the batch of ids
        mask (): the attention mask for the batch
        model   (): the model to use for summarization
        tokenizer (): the tokenizer to use for summarization
        is_general_attention_model (bool, optional): whether the model is a general attention model. Defaults to True.

    Returns:
        str: the summary of the batch
    """

    ids = ids[None, :]
    mask = mask[None, :]

    input_ids = ids.to("cuda") if torch.cuda.is_available() else ids
    attention_mask = mask.to("cuda") if torch.cuda.is_available() else mask

    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    if is_general_attention_model:
        summary_pred_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )
    else:
        # this is for LED etc.
        summary_pred_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )
    summary = tokenizer.batch_decode(
        summary_pred_ids.sequences,
        skip_special_tokens=True,
        remove_invalid_values=True,
    )
    score = round(summary_pred_ids.sequences_scores.cpu().numpy()[0], 4)

    return summary, score


def summarize_via_tokenbatches(
    input_text: str,
    model,
    tokenizer,
    batch_length=4096,
    batch_stride=16,
    **kwargs,
):
    """
    summarize_via_tokenbatches - a function that takes a string and returns a summary

    Args:
        input_text (str): the text to summarize
        model (): the model to use for summarizationz
        tokenizer (): the tokenizer to use for summarization
        batch_length (int, optional): the length of each batch. Defaults to 4096.
        batch_stride (int, optional): the stride of each batch. Defaults to 16. The stride is the number of tokens that overlap between batches.

    Returns:
        str: the summary
    """

    logger = logging.getLogger(__name__)
    # log all input parameters
    if batch_length < 512:
        batch_length = 512
        logging.warning("WARNING: batch_length was set to 512")
    logging.debug(
        f"batch_length: {batch_length} batch_stride: {batch_stride}, kwargs: {kwargs}"
    )
    encoded_input = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=batch_length,
        stride=batch_stride,
        return_overflowing_tokens=True,
        add_special_tokens=False,
        return_tensors="pt",
    )

    in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask
    gen_summaries = []

    pbar = tqdm(total=len(in_id_arr))

    for _id, _mask in zip(in_id_arr, att_arr):

        result, score = summarize_and_score(
            ids=_id,
            mask=_mask,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
        score = round(float(score), 4)
        _sum = {
            "input_tokens": _id,
            "summary": result,
            "summary_score": score,
        }
        gen_summaries.append(_sum)
        print(f"\t{result[0]}\nScore:\t{score}")
        pbar.update()

    pbar.close()

    return gen_summaries


def save_params(
    params: dict,
    output_dir: str or Path,
    hf_tag: str = None,
    verbose: bool = False,
) -> None:
    """
    save_params - save the parameters of the run to a json file

    :param dict params: parameters to save
    :param str or Path output_dir: directory to save the parameters to
    :param str hf_tag: the model tag on huggingface
    :param bool verbose: whether to log the parameters

    :return: None
    """
    output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    session_settings = params
    session_settings["huggingface-model-tag"] = "" if hf_tag is None else hf_tag
    session_settings["date-run"] = get_timestamp()

    metadata_path = output_dir / "summarization-parameters.json"
    logging.info(f"Saving parameters to {metadata_path}")
    with open(metadata_path, "w") as write_file:
        json.dump(session_settings, write_file)

    logging.debug(f"Saved parameters to {metadata_path}")
    if verbose:
        # log the parameters
        logging.info(f"parameters: {session_settings}")
