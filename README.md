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

A continuation of the [document summarization](<https://huggingface.co/spaces/pszemraj/document-summarization>) space on huggingface.

## Installation

```bash
pip install -e .
```

To install all the dependencies _(includes PDF OCR, gradio UI demo)_, run:

```bash
pip install -e .[all]
```

## Usage

### UI Demo

Simply run the following command to start the UI demo:

```bash
ts-ui
```

Other args to be added soon

## Roadmap

- [ ] add argparse CLI for UI demo
- [ ] add CLI for summarization of all text files in a directory
- [ ] API for summarization of text docs

and other things I haven't thought of yet

---

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
