# textsum

 <a href="https://colab.research.google.com/gist/pszemraj/ff8a8486dc3303199fe9c9790a606fff/textsum-summarize-text-files-example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a href="https://pypi.org/project/textsum/"> <img src="https://img.shields.io/pypi/v/textsum.svg" alt="PyPI-Server"/></a>

<br>

> a utility for using transformers summarization models on text docs 🖇

This package provides easy-to-use interfaces for using summarization models on text documents of arbitrary length. Currently implemented interfaces include a python API, CLI, and a shareable demo app.

> [!TIP]
> For additional details, explanations, and docs, see the [wiki](https://github.com/pszemraj/textsum/wiki)

---

- [textsum](#textsum)
  - [🔦 Quick Start Guide](#-quick-start-guide)
  - [Installation](#installation)
    - [Full Installation](#full-installation)
    - [Extra Features](#extra-features)
  - [Usage](#usage)
    - [Python API](#python-api)
    - [CLI](#cli)
    - [Demo App](#demo-app)
  - [Models](#models)
  - [Advanced Configuration](#advanced-configuration)
    - [Parameters](#parameters)
    - [8-bit Quantization \& TensorFloat32](#8-bit-quantization--tensorfloat32)
    - [Using Optimum ONNX Runtime](#using-optimum-onnx-runtime)
    - [Force Cache](#force-cache)
    - [Compile Model](#compile-model)
  - [Contributing](#contributing)
  - [Road Map](#road-map)

---

## 🔦 Quick Start Guide

1. Install the package with pip:

```bash
pip install textsum
```

2. Import the package and create a summarizer:

```python
from textsum.summarize import Summarizer
summarizer = Summarizer() # loads default model and parameters
```

3. Summarize a text string:

```python
text = "This is a long string of text that will be summarized."
summary = summarizer.summarize_string(text)
print(f'Summary: {summary}')
```

---

## Installation

Install using pip with Python 3.8 or later (_after creating a virtual environment_):

```bash
pip install textsum
```

The `textsum` package is now installed in your virtual environment. [CLI commands](#cli) are available in your terminal, and the [python API](#python-api) is available in your python environment.

### Full Installation

For a full installation, which includes additional features such as PDF OCR, Gradio UI demo, and Optimum, run the following commands:

```bash
git clone https://github.com/pszemraj/textsum.git
cd textsum
# create a virtual environment (optional)
pip install -e .[all]
```

### Extra Features

The package also supports a number of optional extra features, which can be installed as follows:

- `8bit`: Install with `pip install -e "textsum[8bit]"`
- `optimum`: Install with `pip install -e "textsum[optimum]"`
- `PDF`: Install with `pip install -e "textsum[PDF]"`
- `app`: Install with `pip install -e "textsum[app]"`
- `unidecode`: Install with `pip install -e "textsum[unidecode]"`

Replace `textsum` in the command with `.` if installing from source. Read below for more details on how to use these features.

> [!TIP]
> The `unidecode` extra is a GPL-licensed dependency not included by default with the `clean-text` package. Installing it should improve the cleaning of noisy input text, but it should not make a significant difference in most use cases.

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

To summarize a directory of text files, run the following command in your terminal:

```bash
textsum-dir /path/to/dir
```

There are many CLI flags available. A full list:


<details>
  <summary>Click to expand table</summary>

  | Flag                             | Description                              |
  | -------------------------------- | ---------------------------------------- |
  | `--output_dir`                   | Specify the output directory             |
  | `--model`                        | Specify the model to use                 |
  | `--no_cuda`                      | Disable CUDA                             |
  | `--tf32`                         | Use TF32 precision                       |
  | `--force_cache`                  | Force cache usage                        |
  | `--load_in_8bit`                 | Load in 8-bit mode                       |
  | `--compile`                      | Compile the model                        |
  | `--optimum_onnx`                 | Use optimum ONNX                         |
  | `--batch_length`                 | Specify the batch length                 |
  | `--batch_stride`                 | Specify the batch stride                 |
  | `--num_beams`                    | Specify the number of beams              |
  | `--length_penalty`               | Specify the length penalty               |
  | `--repetition_penalty`           | Specify the repetition penalty           |
  | `--max_length_ratio`             | Specify the maximum length ratio         |
  | `--min_length`                   | Specify the minimum length               |
  | `--encoder_no_repeat_ngram_size` | Specify the encoder no repeat ngram size |
  | `--no_repeat_ngram_size`         | Specify the no repeat ngram size         |
  | `--early_stopping`               | Enable early stopping                    |
  | `--shuffle`                      | Shuffle the input data                   |
  | `--lowercase`                    | Convert input to lowercase               |
  | `--loglevel`                     | Specify the log level                    |
  | `--logfile`                      | Specify the log file                     |
  | `--file_extension`               | Specify the file extension               |
  | `--skip_completed`               | Skip completed files                     |

</details>


Some useful options are:

Arguments:

- `--model`: model name or path to use for summarization. (Optional)
- `--shuffle`: Shuffle the input files before processing. (Optional)
- `--skip_completed`: Skip already completed files in the output directory. (Optional)
- `--batch_length`: The maximum length of each input batch. Default is 4096. (Optional)
- `--output_dir`: The directory to write the summarized output files. Default is `./summarized/`. (Optional)

To see all available options, run the following command:

```bash
textsum-dir --help
```

### Demo App

For convenience, a UI demo[^1] is provided using [gradio](https://gradio.app/). To ensure you have the dependencies installed, run the following command:

```bash
pip install textsum[app]
```

To launch the demo, run:

```bash
textsum-ui
```

This will start a local server that you can access in your browser & a shareable link will be printed to the console.

[^1]: The demo is minimal but will be expanded to accept other arguments and options.

## Models

Summarization is a memory-intensive task, and the [default model is relatively small and efficient](https://huggingface.co/BEE-spoke-data/pegasus-x-base-synthsumm_open-16k) for long-form text summarization. If you want to use a different model, you can specify the `model_name_or_path` argument when instantiating the `Summarizer` class.

```python
summarizer = Summarizer(model_name_or_path='pszemraj/long-t5-tglobal-xl-16384-book-summary')
```

You can also use the `-m` argument when using the CLI:

```bash
textsum-dir /path/to/dir -m pszemraj/long-t5-tglobal-xl-16384-book-summary
```

Any [text-to-text](https://huggingface.co/models?filter=text2text) or [summarization](https://huggingface.co/models?filter=summarization) model from the [HuggingFace model hub](https://huggingface.co/models) can be used. Models are automatically downloaded and cached in `~/.cache/huggingface/hub`.

---

## Advanced Configuration

### Parameters

Memory usage can also be reduced by adjusting the [parameters for inference](https://huggingface.co/docs/transformers/generation_strategies#beam-search-decoding). This is discussed in detail in the [project wiki](https://github.com/pszemraj/textsum/wiki).

> [!IMPORTANT]
> tl;dr: use the `summarizer.set_inference_params()` and `summarizer.get_inference_params()` methods to adjust the inference parameters, passing either a python `dict` or a JSON file.

Support for `GenerationConfig` as the primary method to adjust inference parameters is planned for a future release.

### 8-bit Quantization & TensorFloat32

Some methods of efficient inference[^2] include loading the model in 8-bit precision via [LLM.int8](https://arxiv.org/abs/2208.07339) (_reduces memory usage_) and enabling TensorFloat32 precision  in the torch backend (_reduces latency_). See the [transformers docs](https://huggingface.co/docs/transformers/perf_infer_gpu_one#efficient-inference-on-a-single-gpu) for more details. Using LLM.int8 requires the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) package, which can either be installed directly or via the `textsum[8bit]` extra:

[^2]: if you have compatible hardware. In general, ampere (RTX 30XX) and newer GPUs are recommended.

```bash
pip install textsum[8bit]
```

To use these options, use the `--load_in_8bit` and `--tf32` flags when using the CLI:

```bash
textsum-dir /path/to/dir --load_in_8bit --tf32
```

Or in Python, using the `load_in_8bit` argument:

```python
summarizer = Summarizer(load_in_8bit=True)
```

If using the Python API, either [manually activate tf32](https://huggingface.co/docs/transformers/perf_train_gpu_one#tf32) or use the `check_ampere_gpu()` function from `textsum.utils` **before initializing the `Summarizer` class**:

```python
from textsum.utils import check_ampere_gpu
check_ampere_gpu() # automatically enables TF32 if Ampere+ available
summarizer = Summarizer(load_in_8bit=True)
```

### Using Optimum ONNX Runtime

> [!CAUTION]
> This feature is experimental and might not work as expected. Use at your own risk. ⚠️🧪

ONNX Runtime is a performance-oriented inference engine for ONNX models. It can be used to increase the speed of model inference, especially on Windows and in environments where GPU acceleration is not available. If you want to use ONNX runtime for inference, you need to set `optimum_onnx=True` when initializing the `Summarizer` class.

First, install with `pip install textsum[optimum]`. Then initialize the `Summarizer` class with ONNX runtime:

```python
summarizer = Summarizer(model_name_or_Path="onnx-compatible-model-name", optimum_onnx=True)
```

It will automatically convert the model if it has not been converted to ONNX yet.

**Notes:**

1. ONNX runtime+cuda needs an additional package. Manually install `onnxruntime-gpu` if you plan to use ONNX with GPU.
2. Using ONNX runtime might lead to different behavior in certain models. It is recommended to test the model with and without ONNX runtime **the same input text** before using it for anything important.

### Force Cache

> [!CAUTION]
> Setting `force_cache=True` might lead to different behavior in certain models. Test the model with and without `force_cache` on **the same input text** before using it for anything important.

Using the cache speeds up autoregressive generation by avoiding recomputing attention for tokens that have already been generated. If you want to force the model to always use cache irrespective of the model's default behavior[^3], you can set `force_cache=True` when initializing the `Summarizer` class.

[^3]: `use_cache` can sometimes be disabled due to things like gradient accumulation training, etc., and if not re-enabled will result in slower inference times.

```python
summarizer = Summarizer(force_cache=True)
```


### Compile Model

If you want to [compile the model](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for faster inference times, you can set `compile_model=True` when initializing the `Summarizer` class.

```python
summarizer = Summarizer(compile_model=True)
```

> [!NOTE]
> Compiling the model might not be supported on all platforms and requires pytorch > 2.0.0.

---

## Contributing

Contributions are welcome! Please open an issue or PR if you have any ideas or suggestions.

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute.

## Road Map

- [x] add CLI for summarization of all text files in a directory
- [x] python API for summarization of text docs
- [ ] add argparse CLI for UI demo
- [x] put on PyPI
- [x] LLM.int8 inference
- [x] optimum inference integration
- [ ] better documentation [in the wiki](https://github.com/pszemraj/textsum/wiki), details on improving performance (speed, quality, memory usage, etc.)
  - [x] in-progress
- [ ] improvements to the PDF OCR helper module (_TBD - may focus more on being a summarization tool_)

_Other ideas? Open an issue or PR!_

---

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
