import logging

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    logging.debug(f"loading model {model_name} to {device}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info(f"Loaded model {model_name} to {device}")
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
    batch_length=2048,
    batch_stride=16,
    **kwargs,
):
    """
    summarize_via_tokenbatches - a function that takes a string and returns a summary

    Args:
        input_text (str): the text to summarize
        model (): the model to use for summarizationz
        tokenizer (): the tokenizer to use for summarization
        batch_length (int, optional): the length of each batch. Defaults to 2048.
        batch_stride (int, optional): the stride of each batch. Defaults to 16. The stride is the number of tokens that overlap between batches.

    Returns:
        str: the summary
    """
    # log all input parameters
    if batch_length < 512:
        batch_length = 512
        print("WARNING: batch_length was set to 512")
    print(
        f"input parameters: {kwargs}, batch_length={batch_length}, batch_stride={batch_stride}"
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
