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

> utility for using transformers summarization models on text docs

An extension/generalization of the [document summarization](<https://huggingface.co/spaces/pszemraj/document-summarization>) space on huggingface. The purpose of this package is to provide a simple interface for using summarization models on text documents of arbitrary length.

⚠️ **WARNING**: _This package is a WIP and is not ready for production use. Some things may not work yet._ ⚠️

## Installation

```bash
git clone https://github.com/pszemraj/textsum.git
cd textsum
# create a virtual environment (optional)
pip install -e .
```

The textsum package is now installed in your virtual environment. You can now use the CLI or UI demo (see [Usage](#usage)).

### Full Installation _(PDF OCR, gradio UI demo)_

To install all the dependencies _(includes PDF OCR, gradio UI demo)_, run:

```bash
pip install -e .[all]
```

## Usage

### CLI

To summarize a directory of text files, run the following command:

```bash
textsum-dir /path/to/dir
```

The following options are available:

```
usage: textsum-dir [-h] [-o OUTPUT_DIR] [-m MODEL_NAME] [-batch BATCH_LENGTH] [-stride BATCH_STRIDE] [-nb NUM_BEAMS]
                   [-l2 LENGTH_PENALTY] [-r2 REPETITION_PENALTY] [--no_cuda] [-length_ratio MAX_LENGTH_RATIO] [-ml MIN_LENGTH]
                   [-enc_ngram ENCODER_NO_REPEAT_NGRAM_SIZE] [-dec_ngram NO_REPEAT_NGRAM_SIZE] [--no_early_stopping] [--shuffle]
                   [--lowercase] [-v] [-vv] [-lf LOGFILE]
                   input_dir
```

For more information, run:

```bash
textsum-dir --help
```

### UI Demo

For convenience, a UI demo is provided using [gradio](https://gradio.app/). To run the demo, run the following command:

```bash
textsum-ui
```

This is currently a minimal demo, but it will be expanded in the future to accept other arguments and options.

---

## Roadmap

- [ ] add argparse CLI for UI demo
- [x] add CLI for summarization of all text files in a directory
- [ ] python API for summarization of text docs
- [ ] optimum inference integration
- [ ] better documentation, details on improving performance (speed, quality, memory usage, etc.)

and other things I haven't thought of yet

---

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
