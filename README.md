<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/textsum.svg?branch=main)](https://cirrus-ci.com/github/<USER>/textsum)
[![ReadTheDocs](https://readthedocs.org/projects/textsum/badge/?version=latest)](https://textsum.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/textsum/main.svg)](https://coveralls.io/r/<USER>/textsum)
[![PyPI-Server](https://img.shields.io/pypi/v/textsum.svg)](https://pypi.org/project/textsum/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/textsum.svg)](https://anaconda.org/conda-forge/textsum)
[![Monthly Downloads](https://pepy.tech/badge/textsum/month)](https://pepy.tech/project/textsum)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/textsum)
-->

# textsum

 <a href="https://colab.research.google.com/gist/pszemraj/ff8a8486dc3303199fe9c9790a606fff/textsum-summarize-text-files-example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a href="https://pypi.org/project/textsum/"> <img src="https://img.shields.io/pypi/v/textsum.svg" alt="PyPI-Server"/></a>

<br>

> utility for using transformers summarization models on text docs

This package provides easy-to-use interfaces for using summarization models on text documents of arbitrary length. Currently implemented interfaces include a python API, CLI, and a shareable demo app.

For details, explanations, and docs, see the [wiki](https://github.com/pszemraj/textsum/wiki)

---

- [textsum](#textsum)
  - [Installation](#installation)
    - [Full Installation](#full-installation)
    - [Additional Details](#additional-details)
  - [Usage](#usage)
    - [Python API](#python-api)
    - [CLI](#cli)
    - [Demo App](#demo-app)
  - [Using Big Models](#using-big-models)
    - [Reducing Memory Usage](#reducing-memory-usage)
      - [EFficient Inference](#efficient-inference)
      - [Parameters](#parameters)
  - [Contributing](#contributing)
  - [Roadmap](#roadmap)

---

## Installation

Install using pip:

```bash
# create a virtual environment (optional)
pip install textsum
```

The `textsum` package is now installed in your virtual environment. CLI commands/python API can summarize text docs from anywhere. see the [Usage](#usage) section for more details.

### Full Installation

To install all the dependencies _(includes PDF OCR, gradio UI demo, optimum, etc)_, run:

```bash
git clone https://github.com/pszemraj/textsum.git
cd textsum
# create a virtual environment (optional)
pip install -e .[all]
```

### Additional Details

This package uses the [clean-text](https://github.com/jfilter/clean-text) python package, and like the "base" version of the package, **does not** include the GPL-licensed `unidecode` dependency. If you want to use the `unidecode` package, install the package as an extra with `pip`:

```bash
pip install textsum[unidecode]
```

In practice, text cleaning pre-summarization with/without `unidecode` should not make a significant difference.

## Usage

There are three ways to use this package:

1. [python API](#python-api)
2. [CLI](#cli)
3. [Demo App](#demo-app)

### Python API

To use the python API, import the `Summarizer` class and instantiate it. This will load the default model and parameters.

You can then use the `summarize_string` method to summarize a long text string.

```python
from textsum.summarize import Summarizer

summarizer = Summarizer() # loads default model and parameters

# summarize a long string
out_str = summarizer.summarize_string('This is a long string of text that will be summarized.')
print(f'summary: {out_str}')
```

you can also directly summarize a file:

```python
out_path = summarizer.summarize_file('/path/to/file.txt')
print(f'summary saved to {out_path}')
```

### CLI

To summarize a directory of text files, run the following command:

```bash
textsum-dir /path/to/dir
```

The following options are available:

```bash
usage: textsum-dir [-h] [-o OUTPUT_DIR] [-m MODEL_NAME] [--no_cuda] [--tf32] [-8bit]
                   [-batch BATCH_LENGTH] [-stride BATCH_STRIDE] [-nb NUM_BEAMS]
                   [-l2 LENGTH_PENALTY] [-r2 REPETITION_PENALTY]
                   [-length_ratio MAX_LENGTH_RATIO] [-ml MIN_LENGTH]
                   [-enc_ngram ENCODER_NO_REPEAT_NGRAM_SIZE] [-dec_ngram NO_REPEAT_NGRAM_SIZE]
                   [--no_early_stopping] [--shuffle] [--lowercase] [-v] [-vv] [-lf LOGFILE]
                   input_dir
```

For more information, run the following:

```bash
textsum-dir --help
```

### Demo App

For convenience, a UI demo[^1] is provided using [gradio](https://gradio.app/). To ensure you have the dependencies installed, clone the repo and run the following command:

```bash
pip install textsum[app]
```

To run the demo, run the following command:

```bash
textsum-ui
```

This will start a local server that you can access in your browser & a shareable link will be printed to the console.

[^1]: The demo is minimal but will be expanded to accept other arguments and options.

## Using Big Models

Summarization is a memory-intensive task, and the [default model is relatively small and efficient](https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary) for long-form text summarization. If you want to use a bigger model, you can specify the `model_name_or_path` argument when instantiating the `Summarizer` class.

```python
summarizer = Summarizer(model_name_or_path='pszemraj/long-t5-tglobal-xl-16384-book-summary')
```

You can also use the `-m` argument when using the CLI:

```bash
textsum-dir /path/to/dir -m pszemraj/long-t5-tglobal-xl-16384-book-summary
```

### Reducing Memory Usage

#### EFficient Inference

Some methods of reducing memory usage _if you have compatible hardware_ include loading the model in 8-bit precision via [LLM.int8](https://arxiv.org/abs/2208.07339) and using the `--tf32` flag to use TensorFloat32 precision. See the [transformers docs](https://huggingface.co/docs/transformers/perf_infer_gpu_one#efficient-inference-on-a-single-gpu) for more details on how this works. Using LLM.int8 requires the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) package, which can either be installed directly or via the `textsum[8bit]` extra:

```bash
pip install textsum[8bit]
```

To use these options, use the `-8bit` and `--tf32` flags when using the CLI:

```bash
textsum-dir /path/to/dir -8bit --tf32
```

Or in python, using the `load_in_8bit` argument:

```python
summarizer = Summarizer(load_in_8bit=True)
```

If using the python API, it's better to initiate tf32 yourself; see [here](https://huggingface.co/docs/transformers/perf_train_gpu_one#tf32) for how.

#### Parameters

Memory usage can also be reduced by adjusting the parameters for inference. This is discussed in detail in the [project wiki](https://github.com/pszemraj/textsum/wiki).

tl;dr for this README, you can use the `.set_inference_params()` and `.get_inference_params()` methods to adjust the parameters for inference.

---

## Contributing

Contributions are welcome! Please open an issue or PR if you have any ideas or suggestions.

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute.

## Roadmap

- [x] add CLI for summarization of all text files in a directory
- [x] python API for summarization of text docs
- [ ] add argparse CLI for UI demo
- [x] put on PyPI
- [x] LLM.int8 inference
- [ ] optimum inference integration
- [ ] better documentation [in the wiki](https://github.com/pszemraj/textsum/wiki), details on improving performance (speed, quality, memory usage, etc.)
- [ ] improvements to the PDF OCR helper module

_Other ideas? Open an issue or PR!_

---

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
