# Phase 1 — Tokenizer Explorer

> **Goal:** Understand how Llama-3.2-3B *sees* text before touching any training code.

---

## What This Phase Is About

Before writing a single line of training code, you need to understand the fundamental unit that the model operates on: **tokens**. Everything downstream — dataset formatting, sequence length limits, batching strategy, memory budgeting — is a direct consequence of how the tokenizer works. Getting this wrong means training on garbage-formatted data and wondering why your model is broken.

This phase is purely exploratory and runs entirely on CPU. No GPU, no model weights loaded. Just the tokenizer (~3MB download).

---

## File

| File | Purpose |
|------|---------|
| [`phase1_tokenizer_explorer.py`](./phase1_tokenizer_explorer.py) | Interactive script that walks through all tokenization concepts with live output |

**Run it:**
```bash
pip install transformers torch huggingface_hub
huggingface-cli login   # paste your HF token
python phase1_tokenizer_explorer.py
```

---

## What I Learned — Section by Section

### Section 0: Loading the Tokenizer
The tokenizer and the model are **separate artifacts**. You can download the tokenizer alone (a few MB) without pulling the 6GB+ model weights. This is the correct workflow for data preparation — you never need the actual model until training time on a GPU machine. This was a subtle but important architectural insight.

### Section 1: What Is a Token?
Tokens are **not words**. The tokenizer uses **Byte-Pair Encoding (BPE)** — a compression algorithm that iteratively merges the most frequent character pairs until it hits a target vocabulary size (128,256 for Llama 3).

Key discoveries:
- Common English words → 1 token each. `"Hello"` = 1 token.
- Rare or compound words get split. `"supercalifragilisticexpialidocious"` = many tokens.
- Numbers are tokenized character-by-character or in small chunks — this is **why LLMs are bad at arithmetic by default**. There's no "number 1234" token; it might be `"12"` + `"34"` or individual digits.
- Non-Latin scripts (Hindi, Chinese) are much more token-expensive than English for the same semantic content. A word in Hindi may cost 3–5× as many tokens.
- Emoji are encoded as UTF-8 byte sequences and can cost multiple tokens each.

**Why this matters for training:** You pay compute cost per token, not per word. If your dataset has lots of numbers, code, or non-English text, your effective dataset size in tokens is much larger than you'd intuitively expect.

### Section 2: Encoding & Decoding (The Round Trip)
The core operation:
```
text → tokenizer.encode() → [list of int IDs] → model
model output [int IDs] → tokenizer.decode() → text
```
Round-trip fidelity is perfect for normal text. The model never sees raw strings — only integer sequences.

### Section 3: Special Tokens
Special tokens are **not** ordinary text. They're reserved IDs that act as signals:

| Token | Meaning |
|-------|---------|
| `<\|begin_of_text\|>` | Start of a new sequence |
| `<\|end_of_text\|>` | Model must stop generating here |

**The key insight for SFT:** Every training sample must end with `<|end_of_text|>`. During training, the model learns to predict this token after completing the assistant's response. Without it, the model has no learned stopping condition and will ramble indefinitely at inference time.

This is why the original project's prompt format appended `{eos}` at the end of every response — it wasn't optional boilerplate, it was a fundamental training signal.

### Section 4: Sequence Length & Training Cost
Transformer attention is **O(n²)** in sequence length. This is the single biggest cost driver in training:

| Max Seq Length | Memory Impact | Viable On |
|---------------|--------------|-----------|
| 512 | Baseline | T4 (15GB), batch_size=4 |
| 1024 | ~4× more than 512 | A100 or gradient checkpointing |
| 2048 | ~16× more than 512 | A100 80GB (original project) |

The original project trained at 2048. For our reproduction run, we'll start at 512 for validation (cheap, fast, proves the pipeline works), then scale up.

**Decision:** `max_seq_length = 512` for the initial Colab validation run.

### Section 5: The Chat Template
Llama-3.2-3B has a **built-in chat template** baked into the tokenizer config. The model was pre-trained to expect a specific conversation format. Training on a different format means the model is learning a mismatched signal — it'll work, but you're leaving performance on the table.

Two modes:
- **Training format** (`add_generation_prompt=False`): Includes the full conversation including the assistant's answer. The model is trained to predict this.
- **Inference format** (`add_generation_prompt=True`): Stops after the user turn. The model continues from there.

**Discovery:** The base model (`Llama-3.2-3B`, not the Instruct variant) may or may not have a chat template. The original project worked around this by using a custom Alpaca-style format:
```
Instruction:
[question]

### Input:
[optional context]

### Response:
[answer]<|end_of_text|>
```
This is a perfectly valid approach — you're defining the format the model will learn during SFT. Consistency is what matters, not which specific format you choose.

---

## Choices & Whys

### Why start with the tokenizer, not the model?
Because every mistake in data formatting is invisible until training fails. If you format your dataset wrong, the model trains silently on garbage. Understanding the tokenizer first means you can verify your formatted samples *before* spinning up an expensive GPU job.

### Why not load the full model?
The model is 6GB+ and needs a GPU to run meaningfully. The tokenizer runs on CPU in under a second. Doing data exploration with the full model is wasteful and slow. This separation — tokenizer on CPU for data prep, model on GPU for training — is the standard professional workflow.

### Why does format consistency matter more than format choice?
The model is a pattern learner. If 95% of your training samples say `### Response:` and 5% say `Assistant:`, the model will learn both but weakly. Consistent formatting means the model learns one strong pattern. What the format *is* matters less than it being the *same* every time.

---

## Reflections

The most surprising thing from this phase: **tokenization is where a lot of "mysterious" LLM behavior is explained**. Bad at math? Because numbers are split into weird token chunks. Struggles with rare names? Because they're fragmented into subword pieces the model rarely saw together. Better at English than other languages? Because English text is far more token-efficient, so the model saw far more English concepts per training token.

This also reframes how you should think about your dataset. It's not "88,031 samples." It's "X million tokens." The token count — not the sample count — is the real unit of training data.

---

## What's Next → Phase 2: Data Pipeline

Load all 10 datasets, write a formatting function for each one that produces the standardized prompt format, and combine them into a single dataset ready for training. The tokenizer knowledge from this phase is directly applied there — every formatted sample gets tokenized to verify it fits within `max_seq_length`.
