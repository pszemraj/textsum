# textsum: Practical Usage Guide

Real examples of what works, what doesn't, and why.

## Quick Start: The Basics

### Install

```bash
pip install textsum
```

For all features:
```bash
pip install textsum[all]  # Includes PDF OCR, Gradio UI, 8-bit, ONNX
```

### Minimal Example

```python
from textsum.summarize import Summarizer

summarizer = Summarizer()
summary = summarizer.summarize_string("Your long text here...")
print(summary)
```

That's it. Default model downloads automatically (first run only, ~570MB).

## Three Ways to Use textsum

### 1. Python API - For Integration

**Summarize a string**:
```python
from textsum.summarize import Summarizer

summarizer = Summarizer()

text = """
Long document text here. Could be scraped from web, extracted from database,
read from file, etc. Length doesn't matter - model will batch it automatically.
"""

summary = summarizer.summarize_string(text)
print(summary)
```

**Summarize a file**:
```python
output_path = summarizer.summarize_file(
    "/path/to/document.txt",
    output_dir="./summaries/"
)
print(f"Summary saved to: {output_path}")
```

**Using the shorthand**:
```python
# These are equivalent
summary = summarizer.summarize_string("text here")
summary = summarizer("text here")

# These are equivalent
output = summarizer.summarize_file("/path/to/file.txt")
output = summarizer("/path/to/file.txt")
```

The `__call__` method is smart - if input looks like a file path and the file exists, it reads the file. Otherwise, treats as text.

### 2. CLI - For Batch Processing

**Process a directory**:
```bash
textsum-dir /path/to/documents/
```

Finds all `.txt` files, summarizes them, saves to `./summarized/` subdirectory.

**Common options**:
```bash
# Specify output directory
textsum-dir ./documents --output_dir ./my_summaries

# Skip already-summarized files (resume interrupted batch)
textsum-dir ./documents --skip_completed

# Use a different model
textsum-dir ./documents --model facebook/bart-large-cnn

# Process PDFs instead of txt
textsum-dir ./pdfs --file_extension pdf

# Shuffle processing order (useful for random sampling)
textsum-dir ./documents --shuffle
```

**Adjust quality vs speed**:
```bash
# Higher quality (slower)
textsum-dir ./docs --num_beams 8 --batch_length 8192

# Faster processing (lower quality)
textsum-dir ./docs --num_beams 2 --batch_length 2048
```

**GPU optimization**:
```bash
# Enable 8-bit quantization (requires bitsandbytes)
textsum-dir ./docs --load_in_8bit

# Enable TensorFloat32 on Ampere+ GPUs
textsum-dir ./docs --tf32

# Compile model (PyTorch 2.0+, first run slow, subsequent fast)
textsum-dir ./docs --compile
```

**Logging**:
```bash
# Save detailed log to file
textsum-dir ./docs --logfile output.log --loglevel 20

# Quiet mode (errors only)
textsum-dir ./docs --loglevel 40
```

### 3. Gradio UI - For Interactive Use

```bash
textsum-ui
```

Opens web interface at `http://localhost:7860` with shareable link.

Features:
- Paste text or upload `.txt`/`.pdf` file
- Adjust parameters with sliders
- Download summary as text file
- Live progress indicator

**Environment variables** (set before launching):
```bash
# Limit input length (words)
export DEMO_MAX_INPUT_WORDS=4096

# Limit PDF pages (to prevent OOM)
export DEMO_MAX_INPUT_PAGES=30

textsum-ui
```

## Model Selection

Default model is `BEE-spoke-data/pegasus-x-base-synthsumm_open-16k`. It's good for most use cases.

### When to Use Different Models

**For scientific papers**:
```python
summarizer = Summarizer(
    model_name_or_path="allenai/led-large-16384-arxiv"
)
```

**For books and long narratives**:
```python
summarizer = Summarizer(
    model_name_or_path="pszemraj/long-t5-tglobal-xl-16384-book-summary"
)
```

**For news articles** (shorter, more factual):
```python
summarizer = Summarizer(
    model_name_or_path="facebook/bart-large-cnn"
)
```

**For multi-lingual** (not well-tested, but works):
```python
summarizer = Summarizer(
    model_name_or_path="csebuetnlp/mT5_multilingual_XLSum"
)
```

Any [summarization](https://huggingface.co/models?filter=summarization) or [text2text-generation](https://huggingface.co/models?filter=text2text-generation) model from HuggingFace should work.

## Tuning Output Quality

### The Three Knobs That Matter Most

**1. Number of beams** - Search width
```python
summarizer = Summarizer(num_beams=4)  # Default
# More beams = better quality, slower
# 2 = fast but rough, 4 = balanced, 8 = high quality
```

**2. Length ratio** - Compression amount
```python
summarizer = Summarizer(max_length_ratio=0.25)  # Default
# 0.25 = output is 25% of input length
# 0.1 = very compressed, 0.5 = barely compressed
```

**3. Repetition penalty** - Creativity vs faithfulness
```python
summarizer = Summarizer(repetition_penalty=2.5)  # Default
# Lower (1.5-2.0) = more conservative, closer to source
# Higher (2.5-3.5) = more creative, risk of drift
```

### Example Configurations

**High-quality, faithful summaries** (for critical documents):
```python
summarizer = Summarizer(
    num_beams=8,
    repetition_penalty=2.0,
    encoder_no_repeat_ngram_size=6,
    length_penalty=1.2,  # Slight bias to longer
)
```

**Fast drafts** (for exploratory work):
```python
summarizer = Summarizer(
    num_beams=2,
    repetition_penalty=3.0,
    token_batch_length=2048,
)
```

**Bullet-point style** (highly compressed):
```python
summarizer = Summarizer(
    max_length_ratio=0.1,  # 10% of original
    num_beams=6,
    length_penalty=0.7,  # Bias to shorter
)
```

**Narrative preservation** (for stories, transcripts):
```python
summarizer = Summarizer(
    max_length_ratio=0.4,  # Less compression
    batch_stride=64,  # More overlap for continuity
    repetition_penalty=2.0,  # Allow more natural language
)
```

## Handling Different Document Types

### Plain Text Files

Just works:
```python
summarizer.summarize_file("document.txt")
```

Text is cleaned with `clean-text` library (removes weird unicode, normalizes whitespace, etc.).

### PDFs with Text Layer

```python
# Extract text first (using your preferred tool)
import pdfplumber
with pdfplumber.open("document.pdf") as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)

summary = summarizer.summarize_string(text)
```

### PDFs without Text (OCR required)

Install PDF extra:
```bash
pip install textsum[PDF]
```

Then:
```python
from textsum.pdf2text import convert_PDF_to_Text

# Convert PDF to text
result = convert_PDF_to_Text("scanned.pdf", max_pages=50)
text = result["converted_text"]

# Summarize
summary = summarizer.summarize_string(text)
```

**Note**: OCR is slow (~5-10 seconds per page). For large PDFs, use `max_pages` to truncate.

### Web Pages

```python
import requests
from bs4 import BeautifulSoup

# Fetch and extract
response = requests.get("https://example.com/article")
soup = BeautifulSoup(response.text, "html.parser")
text = soup.get_text()

# Summarize
summary = summarizer.summarize_string(text)
```

### Word Documents

```python
import docx

doc = docx.Document("file.docx")
text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

summary = summarizer.summarize_string(text)
```

### Markdown

```python
with open("README.md", "r") as f:
    text = f.read()

# Remove code blocks if desired
import re
text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

summary = summarizer.summarize_string(text)
```

## Advanced Usage

### Updating Parameters After Creation

```python
summarizer = Summarizer()

# Change params for next run
summarizer.set_inference_params({
    "num_beams": 8,
    "length_penalty": 1.5,
})

# Run with new params
summary = summarizer.summarize_string(text)
```

Or load from JSON:
```python
summarizer.set_inference_params(config_file="my_params.json")
```

JSON format:
```json
{
    "num_beams": 8,
    "length_penalty": 1.5,
    "repetition_penalty": 2.0,
    "min_length": 10,
    "max_length": 1024
}
```

### Inspecting Batch-Level Results

```python
# Get per-batch summaries with scores
batches = summarizer.summarize_via_tokenbatches(text)

for i, batch in enumerate(batches):
    print(f"Batch {i}:")
    print(f"  Summary: {batch['summary'][0]}")
    print(f"  Score: {batch['summary_score']}")  # Higher = better
    print()
```

Useful for debugging quality issues - you can see which batches are weak.

### Saving Configuration for Reproducibility

```python
# After tuning params, save the config
summarizer.save_config("best_config.json")

# Later, recreate the exact same setup
import json
with open("best_config.json") as f:
    config = json.load(f)

summarizer = Summarizer(
    model_name_or_path=config["model_name_or_path"],
    token_batch_length=config["token_batch_length"],
    batch_stride=config["batch_stride"],
    max_length_ratio=config["max_length_ratio"],
    **config["inference_params"],
)
```

### Custom Post-Processing

```python
# Get summary as string
summary = summarizer.summarize_string(text)

# Apply custom cleanup
summary = summary.replace("In this chapter,", "").strip()
summary = re.sub(r"\s+", " ", summary)  # Collapse whitespace

# Or pass custom phrases to built-in postprocessor
batches = summarizer.summarize_via_tokenbatches(text)
summary = summarizer.save_summary(
    batches,
    return_string=True,
    custom_phrases=["In this section,", "The author discusses,"],
)
```

### Batch Processing with Progress Tracking

```python
from pathlib import Path
from tqdm import tqdm

input_files = list(Path("./documents").glob("*.txt"))
output_dir = Path("./summaries")
output_dir.mkdir(exist_ok=True)

summarizer = Summarizer()

for file in tqdm(input_files, desc="Processing"):
    try:
        output = summarizer.summarize_file(
            file,
            output_dir=output_dir,
        )
        tqdm.write(f"✓ {file.name}")
    except Exception as e:
        tqdm.write(f"✗ {file.name}: {e}")
```

## Performance Optimization

### For CPU

```python
# Use smaller model
summarizer = Summarizer(
    model_name_or_path="facebook/bart-base",  # ~140M params vs 570M
)

# Reduce batch size
summarizer = Summarizer(
    token_batch_length=2048,  # Half default
)

# Fewer beams
summarizer = Summarizer(
    num_beams=2,
)

# Compile (PyTorch 2.0+ only, first run slow)
summarizer = Summarizer(
    compile_model=True,
)
```

**Expected**: ~1000 tokens/minute on modern CPU.

### For GPU

```python
# Use larger model
summarizer = Summarizer(
    model_name_or_path="pszemraj/long-t5-tglobal-xl-16384-book-summary",
)

# Larger batches
summarizer = Summarizer(
    token_batch_length=8192,
)

# Enable TF32 (Ampere+ GPUs: RTX 30XX, A100, etc.)
from textsum.utils import check_ampere_gpu
check_ampere_gpu()  # Auto-enables if compatible

summarizer = Summarizer()
```

**Expected**: ~5000-10000 tokens/minute on modern GPU.

### For GPU with Limited Memory

```python
# 8-bit quantization (requires pip install bitsandbytes)
summarizer = Summarizer(
    load_in_8bit=True,
    # Will automatically use device_map="auto"
)
```

Reduces memory by ~50% with minimal quality loss. Enables running XL models on consumer GPUs.

### Using ONNX (Experimental)

```bash
pip install textsum[optimum]
```

```python
summarizer = Summarizer(
    model_name_or_path="facebook/bart-base",  # Smaller models work better
    optimum_onnx=True,
)
```

**May be faster on CPU**. Test before committing - quality can vary per model.

## Common Issues and Solutions

### Issue: "CUDA out of memory"

**Solutions**:
1. Use 8-bit: `Summarizer(load_in_8bit=True)`
2. Reduce batch size: `Summarizer(token_batch_length=2048)`
3. Use smaller model: `Summarizer(model_name_or_path="facebook/bart-base")`
4. Process on CPU: `Summarizer(use_cuda=False)`

### Issue: Summary is too short/long

**For longer summaries**:
```python
summarizer = Summarizer(
    max_length_ratio=0.4,  # Increase from 0.25
    length_penalty=1.2,    # Bias toward longer
)
```

**For shorter summaries**:
```python
summarizer = Summarizer(
    max_length_ratio=0.15,  # Decrease from 0.25
    length_penalty=0.8,     # Bias toward shorter
)
```

### Issue: Summary is repetitive

```python
summarizer = Summarizer(
    repetition_penalty=3.5,         # Increase from 2.5
    no_repeat_ngram_size=4,         # Increase from 3
    encoder_no_repeat_ngram_size=5, # Increase from 4
)
```

### Issue: Summary misses key points

**Increase context overlap**:
```python
summarizer = Summarizer(
    batch_stride=64,  # Increase from 16
)
```

**Use longer batches**:
```python
summarizer = Summarizer(
    token_batch_length=8192,  # Increase from 4096
)
```

**Increase search quality**:
```python
summarizer = Summarizer(
    num_beams=8,  # Increase from 4
)
```

### Issue: Processing is too slow

**Reduce quality for speed**:
```python
summarizer = Summarizer(
    num_beams=2,              # Decrease from 4
    token_batch_length=2048,  # Decrease from 4096
)
```

**Enable optimizations**:
```python
from textsum.utils import check_ampere_gpu
check_ampere_gpu()  # If Ampere+ GPU

summarizer = Summarizer(
    compile_model=True,  # PyTorch 2.0+
    force_cache=True,    # May help
)
```

### Issue: Different results each run

The default is deterministic (`do_sample=False`). If you're getting different results:

1. Check if you changed `do_sample=True`
2. Model compilation can cause slight numeric differences (usually negligible)
3. Different PyTorch/CUDA versions can have small differences

For strict reproducibility:
```python
import torch
torch.manual_seed(42)

summarizer = Summarizer(
    compile_model=False,  # Avoid compilation
)
# Results should now be identical across runs
```

## Real-World Examples

### Example 1: Summarizing Meeting Transcripts

```python
from textsum.summarize import Summarizer

# Tuned for conversational text
summarizer = Summarizer(
    max_length_ratio=0.2,     # Compress heavily
    repetition_penalty=2.8,    # High to avoid repeating names
    batch_stride=32,           # Good overlap for dialogue
    num_beams=4,
)

# Read transcript
with open("meeting_2024_01_15.txt") as f:
    transcript = f.read()

# Generate summary
summary = summarizer.summarize_string(transcript)

# Save with metadata
output = f"""
# Meeting Summary - 2024-01-15

{summary}

---
Generated with textsum
Model: {summarizer.model_name_or_path}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""

with open("meeting_summary.md", "w") as f:
    f.write(output)
```

### Example 2: Academic Paper Processing

```python
from textsum.summarize import Summarizer
from pathlib import Path
import json

# Use arxiv-specialized model
summarizer = Summarizer(
    model_name_or_path="allenai/led-large-16384-arxiv",
    token_batch_length=8192,
    max_length_ratio=0.15,  # Tight summaries
    num_beams=6,
)

papers_dir = Path("./papers")
summaries = {}

for pdf_path in papers_dir.glob("*.pdf"):
    # Assuming PDFs have text layer
    # (Use pdf2text.convert_PDF_to_Text for scanned papers)
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)

    summary = summarizer.summarize_string(text)
    summaries[pdf_path.stem] = summary

    print(f"Processed: {pdf_path.name}")

# Save all summaries
with open("paper_summaries.json", "w") as f:
    json.dump(summaries, f, indent=2)
```

### Example 3: Book Chapter Summarization

```python
from textsum.summarize import Summarizer

# Use book-tuned model with narrative-friendly settings
summarizer = Summarizer(
    model_name_or_path="pszemraj/long-t5-tglobal-xl-16384-book-summary",
    token_batch_length=8192,
    batch_stride=128,        # High overlap for narrative flow
    max_length_ratio=0.35,   # Less aggressive compression
    repetition_penalty=2.2,  # Lower to preserve authorial voice
    length_penalty=1.1,      # Slight bias to longer for completeness
)

# Read book
with open("book.txt") as f:
    book_text = f.read()

# Split into chapters (assuming "Chapter X" markers)
import re
chapters = re.split(r'\n(?=Chapter \d+)', book_text)

chapter_summaries = []
for i, chapter in enumerate(chapters, 1):
    if len(chapter.strip()) < 100:
        continue  # Skip empty chapters

    print(f"Processing Chapter {i}...")
    summary = summarizer.summarize_string(chapter)
    chapter_summaries.append(f"## Chapter {i}\n\n{summary}")

# Combine all chapter summaries
full_summary = "\n\n".join(chapter_summaries)

with open("book_summary.md", "w") as f:
    f.write("# Book Summary\n\n" + full_summary)
```

### Example 4: Web Content Monitoring

```python
import requests
from bs4 import BeautifulSoup
from textsum.summarize import Summarizer
import schedule
import time

summarizer = Summarizer(
    max_length_ratio=0.15,
    num_beams=2,  # Fast mode
)

def fetch_and_summarize(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract main content (adjust selector for your site)
    content = soup.find("article") or soup.find("main")
    if not content:
        return None

    text = content.get_text(separator="\n", strip=True)
    summary = summarizer.summarize_string(text)

    return summary

def monitor_site():
    url = "https://example.com/news"
    summary = fetch_and_summarize(url)
    if summary:
        # Save or send notification
        with open(f"summary_{time.time()}.txt", "w") as f:
            f.write(summary)
        print(f"New summary generated: {summary[:100]}...")

# Run every hour
schedule.every().hour.do(monitor_site)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Testing Your Configuration

Before processing 1000 documents, test on one:

```python
# Load test document
with open("test_doc.txt") as f:
    test_text = f.read()

# Try configuration
summarizer = Summarizer(
    # Your settings here
)

# Generate summary
summary = summarizer.summarize_string(test_text)

# Evaluate
print("Original length:", len(test_text.split()))
print("Summary length:", len(summary.split()))
print("Compression ratio:", len(summary) / len(test_text))
print("\nSummary:")
print(summary)

# Inspect batch details
batches = summarizer.summarize_via_tokenbatches(test_text)
print(f"\nNumber of batches: {len(batches)}")
for i, batch in enumerate(batches):
    print(f"Batch {i} score: {batch['summary_score']}")
```

Iterate until satisfied, then scale up.

## When NOT to Use textsum

**1. Real-time applications** - Model inference is slow (seconds to minutes). Not suitable for sub-second response requirements.

**2. Extremely long documents** - Multi-thousand page documents may exceed memory limits. Consider chunking at document structure level first (chapters, sections).

**3. Tables and structured data** - Models are trained on prose. Tables, lists, code become mangled. Extract structured data separately.

**4. Multi-document synthesis** - textsum processes documents independently. For "summarize these 50 papers into one meta-summary", you'll need additional logic.

**5. Fine-grained control** - If you need sentence-level alignment, attribution, or query-focused summaries, textsum's batch approach is too coarse.

## Where to Go From Here

**Improve quality**:
- Experiment with different models from HuggingFace
- Fine-tune a model on your domain (see HuggingFace training docs)
- Adjust inference parameters for your content type

**Scale up**:
- Use GPU instances (AWS, GCP, Lambda Labs)
- Parallelize across multiple GPUs (requires custom code)
- Use batch API services (e.g., HuggingFace Inference API)

**Integrate**:
- Build Flask/FastAPI wrapper for HTTP API
- Connect to document management systems
- Add to data pipelines (Airflow, Prefect)

**Extend**:
- Add extractive summarization for comparison
- Implement query-focused summarization
- Build hierarchical summarization (summary of summaries)

The code is straightforward Python - easy to fork and modify for specific needs.
