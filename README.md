# PubMed Character-Level GPT

An educational decoder-only transformer implemented in PyTorch and trained at
the character level on PubMed abstracts. The model uses PyTorch tensor and
neural-network primitives, but does not rely on a high-level transformer
library.

This project explores the mechanics of causal self-attention, multi-head
attention, residual connections, autoregressive training, and token-by-token
generation on biomedical text.

## Model

The default configuration is:

| Component | Value |
|---|---:|
| Context length | 256 characters |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Transformer blocks | 6 |
| Dropout | 0.2 |
| Parameters in the recorded run | 10.81M |

Each decoder block uses pre-LayerNorm, causal multi-head self-attention, a
four-times-expanded feed-forward network, and residual connections.

## Recorded training run

`total.ipynb` contains an exploratory CUDA training run on a corpus collected
from PubMed:

| Step | Training loss | Validation loss |
|---:|---:|---:|
| 0 | 4.7000 | 4.7005 |
| 500 | 1.9169 | 1.9336 |
| 2,500 | 1.1503 | 1.1813 |
| 4,500 | 1.0319 | 1.0694 |

These values describe one learning experiment, not a benchmark or a clinical
evaluation. The model has not been evaluated for factual accuracy.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Build the corpus

```bash
python data/prepare.py
```

The script queries the NCBI E-utilities API for PubMed records matching the
configured medical-informatics and oncology query. It stores the generated
corpus at `data/corpus.txt`; that file is intentionally excluded from Git.

For identifiable API usage, set `NCBI_EMAIL` before running the script. An
optional `NCBI_API_KEY` raises the supported request rate. The script remains
below three requests per second without a key and respects PubMed ESearch's
10,000-record retrieval limit. See the
[NCBI E-utilities usage guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/#chapter2.Usage_Guidelines_and_Requirements).

## Train

```bash
python train.py
```

Training writes `checkpoint.pt`, including the learned weights, character
vocabulary, and model configuration. Checkpoints are excluded from Git because
they can be large.

## Generate text

After training:

```bash
python generate.py
```

The default prompt and generation parameters can be changed at the bottom of
`generate.py`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite checks forward and backward passes, causal masking, generation,
input validation, and structured PubMed XML extraction using a small CPU model.

## Project structure

```text
config.py            default model and training configuration
data/prepare.py      PubMed corpus construction
model.py             decoder-only transformer implementation
train.py             vocabulary creation and training loop
generate.py          checkpoint loading and text generation
tests/               CPU unit tests
total.ipynb           exploratory training notebook and recorded results
```

## Limitations and responsible use

- Character-level generation is computationally inefficient compared with
  subword tokenization.
- The train/validation split is a simple contiguous split of the corpus.
- Generated biomedical text may be incorrect, fabricated, or unsafe.
- This is an educational language-model experiment, not a clinical system and
  not a source of medical advice.
- PubMed abstracts may be copyrighted; users are responsible for following
  NCBI and relevant publisher terms when collecting or redistributing text.
