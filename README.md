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

This package is to provides easy-to-use interfaces for using summarization models on text documents of arbitrary length. Currently implemented interfaces include a python API, CLI, and a shareable demo app.

For details, explanations, and docs, see the [wiki](https://github.com/pszemraj/textsum/wiki)

⚠️ _This is a WIP, but general functionality is available_ ⚠️

---

- [textsum](#textsum)
  - [Installation](#installation)
    - [Full Installation](#full-installation)
    - [Additional Details](#additional-details)
  - [Usage](#usage)
    - [Python API](#python-api)
    - [CLI](#cli)
    - [Demo App](#demo-app)
  - [Contributing](#contributing)
  - [Roadmap](#roadmap)

---

## Installation

Install using pip:

```bash
# create a virtual environment (optional)
pip install textsum
```

The `textsum` package is now installed in your virtual environment. CLI commands/python API can be summarize text docs from anywhere. see the [Usage](#usage) section for more details.

### Full Installation

To install all the dependencies _(includes PDF OCR, gradio UI demo, optimum, etc)_, run:

```bash
git clone https://github.com/pszemraj/textsum.git
cd textsum
# create a virtual environment (optional)
pip install -e .[all]
```

### Additional Details

This package uses the [clean-text](https://github.com/jfilter/clean-text) python package, and like the "base" version of the package **does not** include the GPL-licensed `unidecode` dependency. If you want to use the `unidecode` package, install the package as an extra with `pip`:

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

You can then use the `summarize_string` method to summarize a long string of text.

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

[^1]: The demo is currently minimal, but will be expanded in the future to accept other arguments and options.

---

## Contributing

Contributions are welcome! Please open an issue or PR if you have any ideas or suggestions.

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute.

## Roadmap

- [x] add CLI for summarization of all text files in a directory
- [x] python API for summarization of text docs
- [ ] add argparse CLI for UI demo
- [x] put on pypi
- [ ] optimum inference integration, LLM.int8 inference
- [ ] better documentation [in the wiki](https://github.com/pszemraj/textsum/wiki), details on improving performance (speed, quality, memory usage, etc.)
- [ ] improvements to OCR helper module

_Other ideas? Open an issue or PR!_

---

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
