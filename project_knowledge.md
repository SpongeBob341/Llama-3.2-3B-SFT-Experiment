# Project Knowledge — Llama-3.2-3B SFT

> **Purpose**: This is the single source of truth for everything that has happened, every decision made, and every constraint governing this project. It is a **living document** — update it after every meaningful action: new experiment, changed hyperparameter, discovered bug, breakthrough insight, or completed phase. Both the developer and the AI assistant must read this before doing anything non-trivial.

> **Last Updated**: 2026-07-29 (alignment research session — dataset registry revised, contamination fixes, alignment strategy documented)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Reference Project (Tan's LLMA-O)](#3-reference-project-tans-llma-o)
4. [Constraints and Environment](#4-constraints-and-environment)
5. [Architecture and Design Decisions](#5-architecture-and-design-decisions)
6. [GitHub Progress Logging Convention](#6-github-progress-logging-convention)
7. [Phase Log](#7-phase-log)
8. [Dataset Registry](#8-dataset-registry)
9. [Hyperparameter Registry](#9-hyperparameter-registry)
10. [Experiment Log](#10-experiment-log)
11. [Known Issues and Bugs](#11-known-issues-and-bugs)
12. [Open Questions](#12-open-questions)
13. [Glossary and Key Concepts](#13-glossary-and-key-concepts)
14. [Journal / Session Notes](#14-journal--session-notes)


---

## 1. Project Overview

**Project Name**: LLMAI — AI Assistant with Llama-3.2-3B
**GitHub Corpus**: `SpongeBob341/Llama-3.2-3B-SFT-Experiment`
**Local Path**: `C:\Users\divya\Desktop\Llama-3.2-3B-SFT`

### Goal

Build a **working fine-tuned LLM assistant** using Llama-3.2-3B as the base model. The equally important goal is to **deeply understand every component** — not just reproduce code. Every phase is a learning exercise with documented reasoning for each choice.

### What "SFT" Means Here

**Supervised Fine-Tuning (SFT)**: Take the base Llama-3.2-3B model (which knows language but has no instruction-following behavior) and train it on curated (instruction, response) pairs to teach it how to behave as a helpful assistant.

### Intended End State

- A **fine-tuned, merged model** we produce ourselves (trained on Colab GPU)
- A **FastAPI + vLLM inference server** (to be built, using Tan's server as reference)
- A **CLI chat client** (to be built, using Tan's client as reference)

### What This Project IS vs. IS NOT

| Item | Role |
|------|------|
| `phase1/` | OUR code — Phase 1 complete |
| `phase2/` | OUR code — Phase 2 complete |
| `README.md` | OUR public-facing doc |
| `readme_tan.md` | Reference only — describes Tan's original project |
| `Llama3.2-3B-final-fine-tune.ipynb` | **Reference only** — Tan's training notebook |
| `LLM_Server_Final.py` | **Reference only** — Tan's server implementation |
| `LLM_Client_Final.py` | **Reference only** — Tan's client implementation |
| `requirements.txt` | Reference only — Tan's dependencies |

---

## 2. Repository Structure

```
Llama-3.2-3B-SFT/
|
|-- project_knowledge.md                   <- THIS FILE — master knowledge doc
|-- README.md                              <- Our high-level phases overview (public-facing)
|-- readme_tan.md                          <- Reference project README (Tan's LLMA-O)
|-- requirements.txt                       <- Reference: Tan's Python dependencies
|
|-- .agents/
|   +-- rules/
|       +-- project-knowledge.md           <- Auto-loaded rule: read project_knowledge.md every session
|
|-- phase1/
|   |-- README.md                          <- Our Phase 1 deep-dive notes (tokenizer)
|   +-- phase1_tokenizer_explorer.py       <- Our CPU-only tokenizer exploration script
|
|-- phase2/
|   |-- README.md                          <- Our Phase 2 deep-dive notes (data pipeline)
|   +-- Phase2_Data_Pipeline.ipynb         <- Our Phase 2 data pipeline notebook (Colab)
|
|-- [REFERENCE] Llama3.2-3B-final-fine-tune.ipynb  <- Tan's training notebook
|-- [REFERENCE] LLM_Server_Final.py                <- Tan's FastAPI + vLLM server
|-- [REFERENCE] LLM_Client_Final.py                <- Tan's CLI chat client
```


### Our Files vs. Reference Files

| File | Ours or Reference |
|------|------------------|
| `project_knowledge.md` | OURS |
| `README.md` | OURS |
| `readme_tan.md` | Reference |
| `phase1/` (entire folder) | OURS |
| `phase2/` (entire folder) | OURS |
| `Llama3.2-3B-final-fine-tune.ipynb` | Reference (Tan) |
| `LLM_Server_Final.py` | Reference (Tan) |
| `LLM_Client_Final.py` | Reference (Tan) |
| `requirements.txt` | Reference (Tan) |

---

## 3. Reference Project (Tan's LLMA-O)

> This section documents everything we know about the reference project (authored by Tan, see `readme_tan.md`). We use this as a blueprint to understand what the finished system looks like and to inform our own design choices. We do NOT copy-paste — we understand and reproduce.

### Overview

Tan's project ("LLMA-O") is a completed, working fine-tuned LLM assistant built on `meta-llama/Llama-3.2-3B`. It serves as our primary reference for architecture, training config, and deployment patterns.

### Reference Architecture

**Model**: `meta-llama/Llama-3.2-3B` (base, not Instruct)

**Fine-Tuning**:
- Method: QLoRA (SFT)
- Tool: SFTTrainer (TRL)
- Dataset: 10 datasets, max 10k samples each, final size = **88,031 samples**
- Prompt format (from readme_tan.md):
  ```
  Instruction:
  [Instruction/Question Text]

  ### Input:
  [Optional Context]

  ### Response:
  [Answer]<|end_of_text|>
  ```

**Reference Training Parameters (from readme_tan.md)**:

| Parameter | Value |
|-----------|-------|
| `max_seq_length` | 512 |
| `train_batch_size` (effective) | 32 (via gradient accumulation) |
| `num_train_epochs` | 1 |
| Optimizer | `paged_adamw_32bit` |
| Learning Rate | `2e-4` with cosine scheduler |
| Training Environment | ASU cluster, NVIDIA A100 80GB |
| Training Time | ~2 hours 20 minutes |

**Reference Training Metrics**:

| Metric | Value |
|--------|-------|
| `train_loss` | 1.5376 |
| `train_runtime` | 8087.89 seconds |
| `train_samples_per_second` | 10.884 |
| `eval_samples` | 800 |
| `total_flos` | 456,366,752,067,889,150 |

**Reference Inference**:
- Engine: vLLM (PagedAttention)
- Self-Consistency: 5 samples, ROUGE-L similarity majority voting
- Inference speed: 27,500 prompts in ~17 minutes (vs ~3 hours with HF pipeline)

**Reference Server**:
- FastAPI, port 8004
- Stores conversation context
- File: `LLM_Server_Final.py`

**Reference Client**:
- Python CLI
- File: `LLM_Client_Final.py`

### Techniques Explored in Reference (What NOT to Repeat Blindly)

| Technique | Outcome |
|-----------|---------|
| DPO (Direct Preference Optimization) | Attempted with ultrafeedback dataset. Model failed basic interactions post-training. Abandoned. |
| Few-Shot Prompting (static + dynamic) | Ineffective or detrimental. Dynamic SFT-model classifier performed poorly. |
| Chain-of-Thought Prompting | Made answers unnecessarily verbose. Math datasets (gsm8k, metamathqa) already enabled good reasoning. |
| ROUGE-L + Math-based majority voting | Incorrect identification of math questions. |
| Cosine similarity majority voting | Frequently chose hallucinated answers. |
| **ROUGE-L similarity majority voting** | **Best result — the chosen method** |

### Key Note on Prompt Format Discrepancy

`readme_tan.md` shows `### Input:` and `### Response:` (with `###`). However, Tan's server code (`LLM_Server_Final.py`) uses `Input:` and `Response:` **without** `###`. We must confirm which format our Phase 2 notebook actually uses and ensure training and inference formats match exactly. This is tracked as bug B001.

---

## 4. Constraints and Environment

### Hardware Constraints

| Resource | Situation | Impact |
|----------|-----------|--------|
| Local machine (Divya's) | CPU only, no GPU | No training locally; all GPU work on Colab |
| Google Colab Free | T4 (15GB VRAM), session limits | Can run validation; tight on memory |
| Google Colab Pro/A100 | A100 80GB | Needed to match Tan's batch_size=32 at seq_len=512 |
| Reference environment | ASU cluster, A100 80GB | How Tan trained the final model |

### Software Environment (Reference, from requirements.txt)

- `torch`, `transformers`, `datasets`, `accelerate`
- `peft` — LoRA adapters
- `bitsandbytes` — 4-bit quantization
- `trl` — SFTTrainer
- `sentencepiece` — tokenizer
- `evaluate`, `bert_score`, `rouge_score`, `nltk` — evaluation
- `fastapi`, `uvicorn` — API server
- `vllm` — inference engine (**Linux-only**)
- `requests`, `numpy`, `tensorboard`

### Key Constraints

- `max_seq_length = 512` — reference used 512 even on A100
- Samples exceeding `max_seq_length` are **filtered out**, not truncated
- Max 10,000 samples per dataset
- vLLM is Linux-only — local Windows machine cannot run the server
- `meta-llama/Llama-3.2-3B` requires HuggingFace login and model access approval

---

## 5. Architecture and Design Decisions

### 5.1 Why Llama-3.2-3B?

- Colab-accessible at 4-bit quantization (fits T4 15GB VRAM)
- Large enough for meaningful instruction following
- Fast iteration cycles (smaller model = faster training epochs)
- All techniques (QLoRA, SFT, vLLM) transfer to larger models — this is the learning vehicle

### 5.2 Why QLoRA (not full fine-tuning)?

Full fine-tuning of 3B parameters needs ~24GB+ VRAM for weights alone plus optimizer states. Exceeds even A100 40GB for this use case.

**QLoRA approach**:
1. Load model in 4-bit NF4 quantization (`bitsandbytes`) — drastically reduces weight memory
2. Inject LoRA adapters (`peft`) into attention projection layers — only adapters are trained
3. Fine-tuning a 3B model on ~15GB VRAM becomes feasible

**Trade-off**: Marginally lower quality than full fine-tuning. Acceptable because SFT for instruction following is exactly where QLoRA performs well.

### 5.3 Why 10 Datasets?

One dataset = one skill. The 10 datasets cover distinct capability axes:

| Axis | Datasets |
|------|----------|
| Instruction following | Alpaca, Dolly |
| Safety | HH-RLHF |
| Math reasoning | GSM8K, MetaMathQA |
| Multi-hop reasoning | HotpotQA |
| Broad knowledge | MMLU |
| Common sense | CommonsenseQA |
| Factual recall | TriviaQA |
| Truthfulness | TruthfulQA |

### 5.4 Prompt Format (Our Choice)

Based on the reference project and our Phase 2 implementation. The format used (confirm from Phase 2 notebook):

```
Instruction:
{instruction text}

Response:
{answer}<|end_of_text|>
```

With optional context:

```
Instruction:
{instruction text}

Input:
{context}

Response:
{answer}<|end_of_text|>
```

**Why NOT the Llama-3 built-in chat template?**
- We fine-tune the **base model**, not the Instruct variant
- The base model may lack a baked-in chat template
- Alpaca-style is well-established, simple, and consistent
- Consistency of format matters more than which format is chosen

**Why `<|end_of_text|>` is mandatory**:
- It is the model's learned stop signal
- Without it during training, the model produces no stopping condition at inference — it rambles indefinitely

### 5.5 Why max_seq_length = 512?

Transformer attention is O(n²) in sequence length:
- 512 tokens → baseline
- 1024 → ~4× memory
- 2048 → ~16× memory

512 is the validated sweet spot for our hardware budget. Reference trained at 512 and achieved good results.

### 5.9 Alignment Strategy — Two-Stage Pipeline: SFT (Skills) → ORPO (Alignment)

**Decision**: Adopt a **Two-Stage Pipeline** focusing on **Quality > Quantity**:
1. **Stage 1 (SFT)**: High-quality instruction following, math, and multi-hop reasoning.
2. **Stage 2 (ORPO)**: Alignment, safety, and truthfulness using natural preference pairs `(chosen, rejected)`.

**Reasoning & Risk Mitigation for ORPO**:
- **Why ORPO in Stage 2**: Directly aligns the model on preference boundaries without needing a separate reward model or complex PPO setup.
- **Mitigating ORPO's internal SFT loss overlap**: To prevent Stage 2 ORPO's internal SFT component from overwriting Stage 1 skills, Stage 2 will use **ultra-high quality `chosen` responses** (e.g. UltraFeedback, HH-RLHF, HaluEval) with a low learning rate and tuned odds-ratio penalty ($\lambda$).
- **Quality over Quantity**: Instead of blending 88k medium-quality samples, we curate high-signal, top-tier datasets (~25k-40k for SFT, ~15k-25k preference pairs for ORPO).

**What NOT to do**:
- Do not include evaluation benchmarks (MMLU, CommonsenseQA, TruthfulQA) in training data (prevents benchmark contamination).
- Do not use low-quality synthetic responses in ORPO `chosen` pairs.

### 5.6 Why vLLM for Inference?

- Plain `model.generate()`: ~1-3 tokens/sec, no batching, Python GIL
- vLLM with PagedAttention: continuous batching, ~10-100× throughput improvement
- Reference confirmed: 27,500 prompts in 17min with vLLM vs ~3 hours with HF pipeline

**Constraint**: vLLM is Linux-only — cannot run on local Windows machine.

### 5.7 Why ROUGE-L Self-Consistency?

Generate 5 candidates, pick the one with highest average pairwise ROUGE-L similarity (most "consensus" answer).

- ROUGE-L measures LCS (Longest Common Subsequence) — rewards structural similarity over exact match
- Better than majority voting for open-ended text (no two answers are identical)
- Reference tested 3 methods and ROUGE-L only won

**Parameters**: n=5, temperature=0.7, top_p=0.9, max_new_tokens=512

### 5.8 Model Merging

After QLoRA training: merge LoRA adapter weights back into base model weights → produce a single standalone model.
- Required because vLLM cannot natively load PEFT adapters in all versions
- Output model path convention: `llama3.2-3b-sft-merged-{config}` (name TBD for our run)

---

## 6. GitHub Progress Logging Convention

> **Rule**: Every phase of this project gets its own folder in the repository with two required files. This is how progress is logged to GitHub for every phase.

### Folder Structure Per Phase

```
phaseN/
├── README.md                         ← Understanding, reflections, choices, whys, discoveries
└── phaseN_<descriptive_name>.py      ← The main code for that phase (or .ipynb for notebooks)
```

**Example (Phase 1)**:
```
phase1/
├── README.md
└── phase1_tokenizer_explorer.py
```

### The Two Required Files

#### 1. `phaseN/README.md` — The Reflection Document
This is the heart of the logging system. It should contain:

| Section | Content |
|---------|---------|
| **What This Phase Is About** | 1-paragraph context — why this phase exists and what it unlocks for future phases |
| **Files** | Table of files in the folder and their purpose, with run instructions |
| **What I Learned — Section by Section** | Key learnings for each section/component of the code, written in plain language |
| **Choices & Whys** | Every non-obvious decision made during this phase — what the alternatives were and why this path was taken |
| **Reflections** | Surprising discoveries, "aha" moments, things that reframed the understanding of the problem |
| **What's Next** | Brief 2–3 bullet preview of what the next phase will build on from this one |

**Tone**: Written as if explaining to a future self who has forgotten everything. Not a bullet list of facts — real sentences with reasoning.

#### 2. `phaseN/<code_file>` — The Phase Code
- Self-contained script or notebook for that phase
- Heavily commented — every section explains **why** the code works the way it does, not just **what** it does
- Should be runnable independently with a clear setup section at the top

### Root `README.md` — The Living Plan

The root `README.md` contains a **🗺️ Learning Journey & Plan** section at the top. This section:

- Has a phase table showing all phases, their folders, topics, and `✅ Done` / `🔜 Next` / `⬜ Planned` status
- Has a **Current Plan** subsection describing the next phase in 3–5 bullet points
- Is **only updated when there is a major change** in direction, scope, or the overall plan — not after every small tweak

### Workflow for Each New Phase

When starting a new phase `N`:

1. **Create the folder**: `phaseN/`
2. **Write the code first** → `phaseN/phaseN_<name>.py` (or `.ipynb`)
3. **After completing the phase**, write `phaseN/README.md` with full reflections — don't write the README before doing the work
4. **Update root `README.md`**: mark phase N as `✅ Done`, set phase N+1 to `🔜 Next`, update the Current Plan section
5. **Update `project_knowledge.md`**: mark phase N as `[COMPLETE]` in the Phase Log, add a Journal entry
6. **Git commit**: `git add phase{N}/ README.md project_knowledge.md && git commit -m "Phase N: <Topic> complete"`

### Status Badges for Phase Table

| Badge | Meaning |
|-------|---------|
| ✅ Done | Phase complete — folder and README exist |
| 🔜 Next | This is the current active phase |
| ⬜ Planned | Not yet started |
| 🚧 In Progress | Started but not complete |

---

## 7. Phase Log


### Phase 1 — Tokenizer Explorer [COMPLETE]

**Folder**: `phase1/`
**Environment**: CPU only (local machine)
**Goal**: Understand tokenization before writing any training code

**Key Learnings**:
- Tokens are not words — BPE (Byte-Pair Encoding), vocab size = 128,256 for Llama-3
- Numbers tokenized character-by-character — explains LLM arithmetic failures
- Non-Latin scripts are 3-5x less token-efficient than English
- `<|end_of_text|>` is a critical training signal — model learns to stop generating here
- Tokenizer and model are **separate artifacts** — tokenizer (~3MB) runs on CPU without the model (~6GB)
- Attention is O(n²) in sequence length — the single biggest training cost driver

**Decision Locked In**: `max_seq_length = 512` for our training run.

---

### Phase 2 — Data Pipeline [COMPLETE]

**Folder**: `phase2/`
**Environment**: Colab CPU / CPU only
**Goal**: Load all 8 selected datasets, write custom formatters, exclude benchmarks, filter out sequence lengths > 512, combine, and split/push to HuggingFace Hub
**Status**: Complete. Notebook formatted and refactored into the `phase2/` directory along with a reflecting README. SFT and ORPO datasets generated.

**Config Used**:
```
SMALL_RUN   = True         # validation mode
MAX_SAMPLES = 3,000        # per dataset (small run); 10,000 for full run
MAX_SEQ_LEN = 512          # filter out samples longer than this
```

**Key Learnings**:
- Every dataset has a different schema — custom formatters per dataset are mandatory
- Filter (remove) vs. truncate: truncation destroys the completion portion of training samples
- Dataset analysis (token length distribution) before training is critical
- Several originally planned datasets (MMLU, CommonsenseQA, TruthfulQA) are standard eval benchmarks — training on them causes benchmark contamination and must be avoided
- Dataset saved and pushed to HuggingFace Hub for loading during training (NOT YET DONE)

**Output**: Not yet pushed. Hub repo name TBD — update Q001 when pushed.

---

### Phase 3 & 4 — QLoRA Setup + SFT Training Loop [✅ DONE]

**Folder**: `phase3/`
**Environment**: Colab GPU (T4, tested; A100 recommended for full run)
**Artifact**: `SpongeBob387/llama3.2-3b-sft-stage1` (private HF Hub repo)

**Final test-run results** (T4, 11k samples, 1 epoch):
- Train Loss: **1.1717**
- Eval Loss (best): **1.1367** (step 343, end of epoch)
- Runtime: **64.8 minutes** (3,889 seconds) on T4
- Inference quality: ✅ All 3 sanity checks passed (conceptual, math/GSM8K format, factual QA)

**Architecture decision**: Using **Unsloth** over raw HuggingFace stack. Reason: 2-5x faster training, 60% VRAM reduction — makes T4 viable. Produces identical model artifact.

**Confirmed config** (T4 test run):
- `lora_r = 16` (T4; use r=64 on A100 to match Tan)
- `lora_alpha = 16`, `lora_dropout = 0` (Unsloth requirement)
- `target_modules`: q, k, v, o, gate, up, down (7 layers)
- `max_grad_norm = 0.3`, `group_by_length = True`
- `eval_strategy = "steps"`, `eval_steps = 200`
- Padding-free batching auto-enabled by Unsloth (more efficient than padding)

**Bugs encountered and resolved** (see B005, B006 in bug log):
- B005: Unsloth fused CE shape mismatch — fixed with standard CE + min-size alignment
- B006: TRL SFTConfig migration — fixed by using SFTConfig instead of TrainingArguments

---

### Phase 4 — Training Loop [PLANNED]

**Goal**: Run SFTTrainer on our combined dataset

**Reference config** (from Tan):
- `max_seq_length = 512`
- `num_train_epochs = 1`
- `effective_batch_size = 32` (per_device_batch_size x gradient_accumulation_steps)
- `learning_rate = 2e-4`
- `lr_scheduler = cosine`
- `optimizer = paged_adamw_32bit`
- Expected training time: ~2h20min on A100 80GB

---

### Phase 5 — Inference and Evaluation [PLANNED]

**Goal**: Load merged model, run inference, evaluate

**Evaluation targets**:
- ROUGE score
- BERTScore
- Qualitative assessment on a diverse prompt set

---

### Phase 6 — FastAPI Server and Client [PLANNED]

**Goal**: Build our own server and client (using Tan's as reference, not copying)
**Reference files**: `LLM_Server_Final.py`, `LLM_Client_Final.py`

---

## 9. Hyperparameter Registry

> Update this section every time a hyperparameter is set, confirmed, or changed.

### Data / Tokenizer

| Parameter | Value | Source | Why |
|-----------|-------|--------|-----|
| `max_seq_length` | 512 | Locked in Phase 1 | Memory budget; O(n²) attention cost |
| `MAX_SAMPLES` | 3000 (small) / 10000 (full) | Phase 2 | Per-dataset cap to prevent imbalance |
| Filtering strategy | Filter out (not truncate) | Phase 2 | Truncation destroys completions |

### Training (Reference Values — to be confirmed and reproduced)

| Parameter | Reference Value | Our Value | Why |
|-----------|----------------|-----------|-----|
| `load_in_4bit` | True | True | QLoRA memory reduction — ~6GB model → ~1.8GB VRAM |
| `bnb_4bit_quant_type` | `nf4` | `nf4` (via Unsloth) | NF4 optimal for normally distributed weights |
| `bnb_4bit_compute_dtype` | `bfloat16` | auto (Unsloth) | Unsloth auto-detects bf16/fp16 |
| `bnb_4bit_use_double_quant` | True | internal (Unsloth) | Unsloth handles double quant internally |
| `lora_r` | **64** (A100) | **16** (T4) | Higher rank needs more VRAM; r=16 safe for T4 |
| `lora_alpha` | 16 | 16 | Scaling factor; alpha=r gives scale=1.0 |
| `lora_dropout` | 0.1 | 0 | Unsloth requires 0; non-zero triggers HF fallback |
| `target_modules` | q,k,v,o,gate,up,down | q,k,v,o,gate,up,down | All 7 projection layers targeted |
| `learning_rate` | `2e-4` | `2e-4` | Standard for LoRA SFT |
| `num_train_epochs` | `1` | `1` | Reference ran 1 epoch |
| Effective batch size | `32` | `32` (4×8 grad accum) | Via gradient accumulation on T4 |
| `lr_scheduler_type` | `cosine` | `cosine` | Smooth LR decay after warmup |
| `optimizer` | `paged_adamw_32bit` | `paged_adamw_32bit` | Memory-efficient optimizer |
| `max_grad_norm` | `0.3` | `0.3` | Gradient clipping — prevents explosion |
| `weight_decay` | `0.001` | `0.001` | L2 regularization |
| `warmup_ratio` | `0.03` | `0.03` | First 3% of steps ramp LR from 0 |
| `group_by_length` | True | True | Groups similar-length samples for efficiency |

### Inference (Reference Values)

| Parameter | Value | Why |
|-----------|-------|-----|
| `n` samples | 5 | Self-consistency candidates |
| `temperature` | 0.7 | Introduces response diversity |
| `top_p` | 0.9 | Nucleus sampling |
| `max_new_tokens` | 512 | Answer length cap |
| `tensor_parallel_size` | 1 | Single GPU |
| `gpu_memory_utilization` | 0.95 | Leave 5% for OS |
| Server port | 8004 | Reference default |

---

## 10. Experiment Log

> Log every run here: config, outcome, learnings. Never delete old entries.

### Experiment 001 — Phase 1: Tokenizer Exploration

- **Date**: Before 2026-07-29
- **File**: `phase1/phase1_tokenizer_explorer.py`
- **Environment**: Local CPU
- **Outcome**: Verified BPE behavior, special tokens, O(n²) attention cost, chat template behavior, CPU-only tokenizer workflow.
- **Status**: Complete

### Experiment 002 — Phase 2: Data Pipeline (Small Run)

- **Date**: Before 2026-07-29
- **File**: `Phase2_Data_Pipeline.ipynb`
- **Config**: SMALL_RUN=True, MAX_SAMPLES=3000, MAX_SEQ_LEN=512
- **Outcome**: Started mapping datasets. Re-evaluating dataset selections for contamination.
- **Status**: IN PROGRESS

---

## 11. Known Issues and Bugs

| ID | Description | File | Status | Notes |
|----|-------------|------|--------|-------|
| B001 | **Prompt format discrepancy**: `readme_tan.md` uses `### Input:` / `### Response:` but Tan's server code uses `Input:` / `Response:` without `###`. | Cross-cutting | **RESOLVED** | Our Phase 2 and Phase 3 both use `### Input:` / `### Response:`. The discrepancy is with Tan's *server* only — not within our own project. Our training and inference formats match. |
| B002 | vLLM is Linux-only. Cannot test server on local Windows machine. | Deployment | By design | Server must run on Colab/Linux |
| B003 | `@app.on_event("startup")` is deprecated in newer FastAPI versions | Tan's `LLM_Server_Final.py` | Reference only | Use `lifespan` context manager when we build ours |
| B004 | Response cleaning in Tan's client is brittle (hardcoded prefix check) | Tan's `LLM_Client_Final.py` | Reference only | Build ours more robustly |
| B005 | **Unsloth fused CE shape mismatch**: `unsloth_zoo >= 2026.8` + `TRL >= 1.10.0` + newer Torch produces `ValueError: Expected input batch_size (2044) to match target batch_size (2048)`. Caused by misaligned chunking of hidden_states vs labels in `accumulate_chunk()`. | `phase3/Phase3_Stage1_QLoRA_SFT.ipynb` | **RESOLVED** | Replace `unsloth_fused_ce_loss` with standard `F.cross_entropy` + `min(logits.size(1), labels.size(1))` alignment. Patch applied as Cell 5 in the notebook. |
| B006 | **TRL SFTConfig migration**: `TRL >= 0.10.0` deprecated passing `dataset_text_field`, `max_seq_length`, `packing` directly to `SFTTrainer`. These args must now go into `SFTConfig` (which replaces `TrainingArguments` for SFT). | `phase3/Phase3_Stage1_QLoRA_SFT.ipynb` | **RESOLVED** | Notebook updated to use `SFTConfig` exclusively. `SFTConfig` inherits `TrainingArguments` and adds all SFT-specific fields. |

---

## 12. Open Questions

| # | Question | Priority | Context |
|---|----------|----------|---------|
| Q001 | What is the exact HuggingFace Hub repo name for our Phase 2 dataset? | **ANSWERED** | `SpongeBob387/llama3.2-3b-sft-dataset` — confirmed live during Phase 3 test run. Train: 10,974 / Eval: 224 samples. |
| Q002 | What are Tan's exact LoRA hyperparameters (r, alpha, target_modules)? | **ANSWERED** | r=64, alpha=16, dropout=0.1, target all 7 proj layers. We use r=16/dropout=0 for T4. |
| Q003 | Which GPU will we use for training — Colab Free T4 or Pro A100? | High | Determines whether r=16 (T4) or r=64 (A100) is used. |
| Q004 | Does our Phase 2 prompt use `### Input:` or just `Input:`? | **ANSWERED (B001 RESOLVED)** | Uses `### Input:` and `### Response:`. Matches Phase 3 inference format. |
| Q005 | Are we using the small-run dataset (~18k) or full dataset (~35-45k)? | Medium | Impacts training time and compute cost |
| Q006 | What model checkpoint name will we give our merged model? | Medium | For server configuration |
| Q007 | Will we push our fine-tuned model to HuggingFace Hub? | Medium | Convenience for deployment |
| Q008 | What is the `total_flos` of our training run? | Low | Sanity check vs. reference 274T FLOPs (Tan's MIG GPU run) |
| Q009 | Does the Phase 2 notebook currently include MMLU, CommonsenseQA, TruthfulQA formatters? | **ANSWERED** | Phase 2 README confirms these are strictly excluded (benchmark quarantine). |
| Q010 | Why exactly did Tan's DPO attempt fail? | Medium | Must understand before attempting Phase 4.5 ORPO |

---

## 14. Glossary and Key Concepts

| Term | Definition |
|------|------------|
| **SFT** | Supervised Fine-Tuning — training a base model on (instruction, response) pairs to teach assistant behavior |
| **QLoRA** | Quantized Low-Rank Adaptation — 4-bit quantized base model + small trainable LoRA adapters |
| **LoRA** | Low-Rank Adaptation — small trainable matrices injected into frozen model layers |
| **BPE** | Byte-Pair Encoding — tokenization algorithm; iteratively merges most frequent character pairs |
| **NF4** | Normal Float 4 — 4-bit quantization format optimal for normally distributed weight tensors |
| **BitsAndBytes** | CUDA quantization library; enables 4-bit/8-bit model loading |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella for methods like LoRA that train only a small parameter subset |
| **SFTTrainer** | HuggingFace TRL trainer purpose-built for SFT; handles packing, loss masking, etc. |
| **vLLM** | High-throughput LLM inference engine using PagedAttention for memory-efficient serving |
| **PagedAttention** | vLLM's KV-cache memory scheme — non-contiguous pages for efficient GPU memory utilization |
| **Self-Consistency** | Generate N candidates, pick the most "consensus" one — reduces hallucination and noise |
| **ROUGE-L** | Longest Common Subsequence F1 score — measures structural similarity; used for self-consistency selection |
| **Alpaca Format** | `Instruction:\n{x}\n\nResponse:\n{y}<eos>` — the prompt format used throughout this project |
| **Chat Template** | Llama's built-in conversational format; not used here since we fine-tune the base (non-Instruct) model |
| **Merging** | Post-training: fold LoRA adapter weights into base model weights for standalone deployment |
| **O(n²)** | Attention complexity in sequence length — doubling length quadruples compute cost |
| **BERTScore** | Semantic similarity metric using BERT embeddings; more meaning-aware than ROUGE |
| **HH-RLHF** | Helpful + Harmless Reinforcement Learning from Human Feedback — Anthropic's safety dataset |
| **DPO** | Direct Preference Optimization — tried by Tan, abandoned (model broke post-training) |
| **paged_adamw_32bit** | Memory-efficient Adam optimizer variant designed for QLoRA training |
| **gradient_accumulation_steps** | Accumulate gradients over N micro-steps before one weight update — simulates larger batch sizes |

---

## 14. Journal / Session Notes

### 2026-07-29 — Initial Document Creation and Project Audit

**Session goal**: Create `project_knowledge.md` as the master living document.

**Work done**:
- Audited entire project directory
- Read `README.md`, `phase1/README.md`, `readme_tan.md`
- Read all reference files: `LLM_Server_Final.py`, `LLM_Client_Final.py`, `Phase2_Data_Pipeline.ipynb` header
- Extracted all training parameters from `readme_tan.md`
- Clarified which files are OURS vs. which are REFERENCE

**Key clarification from user**:
- `Llama3.2-3B-final-fine-tune.ipynb`, `LLM_Server_Final.py`, `LLM_Client_Final.py` are **Tan's reference files** — not our own project code
- `readme_tan.md` = Tan's original project README
- We build our own training notebook, server, and client, using these as reference

**State of project at this session**:
- Phase 1 (Tokenizer): Complete
- Phase 2 (Data Pipeline): IN PROGRESS — still researching and brainstorming, nothing pushed to Hub
- Phase 3 (QLoRA Setup): Not started
- Phase 4 (Training Loop): Not started
- Phase 5 (Evaluation): Not started
- Phase 6 (Server + Client): Not started

**Critical open item**: Must read Tan's training notebook (`Llama3.2-3B-final-fine-tune.ipynb`) to extract exact LoRA config and confirm the prompt format used during training. This is needed before starting Phase 3.

**Next immediate actions**:
1. Read `Llama3.2-3B-final-fine-tune.ipynb` and fill in Section 9 (Hyperparameter Registry)
2. Confirm exact prompt format in our Phase 2 notebook — resolve B001
3. Begin Phase 3 — write our own QLoRA training notebook
4. Update this document after each step

---

### 2026-07-29 — Adopted Two-Stage Pipeline (SFT → ORPO) with Quality-First Datasets

**Session Goal**: Finalize pipeline direction based on user preference for **Two-Stage SFT → ORPO** and **Quality > Quantity** dataset curation.

**Key Decisions & Changes**:

1. **Pipeline Choice**: Locked in **Stage 1 (QLoRA SFT)** for capability & skill acquisition, followed by **Stage 2 (QLoRA ORPO)** for safety, alignment, and truthfulness.
2. **Quality > Quantity Dataset Strategy**:
   - **Stage 1 SFT (~35k-45k total)**: Swapped out lower-quality synthetic data (Alpaca) for **OpenHermes-2.5** (GPT-4 tier), alongside Dolly-15k, GSM8K, MetaMathQA, and HotpotQA.
   - **Stage 2 ORPO (~20k preference pairs)**: Uses native `(prompt, chosen, rejected)` datasets: **UltraFeedback Binarized** (helpfulness/quality), **HH-RLHF Harmless** (safety/refusals), and **HaluEval** (hallucination mitigation).
3. **ORPO Safety Net**:
   - Stage 2 learning rate will be conservative (`5e-5` to `1e-4`).
   - High-quality `chosen` samples used in ORPO ensure the internal SFT loss reinforces quality while the odds-ratio penalty enforces safety boundaries.

---

### 2026-07-29 — GitHub Progress Logging Convention Added + Document Hardened

**Session goal**: Refine `project_knowledge.md` — incorporate the GitHub progress logging convention, fix section numbering, and make the document more robust.

**Changes made**:
- User added **Section 6: GitHub Progress Logging Convention** defining the per-phase folder structure, the two required files (phase README + code), the root README living plan, the step-by-step phase workflow, and status badge semantics
- User added `.agents/rules/project-knowledge.md` to the repo structure — an auto-loaded rule that instructs the AI assistant to read `project_knowledge.md` at the start of every session
- Section numbers in TOC were updated (shifted from 6–13 to 7–14 to accommodate the new section 6)
- Section numbers in body headers were corrected to match TOC (fixed the duplicate `## 7.` and renumbered 8–14)
- User clarified: `project_knowledge.md` is **flexible** — update it when there are better findings, improved plans, or new information. Do not treat it as frozen.

**Key policy confirmed**:
- This document is a **living document** — AI assistant should update it proactively, not just append
- If a better approach is found, update the relevant section and journal it — don't leave outdated info

**State of project at this session**: Unchanged from previous entry (Phases 1–2 complete, 3–6 not started).

**Next immediate actions**: Same as previous entry — read Tan's training notebook, resolve B001, begin Phase 3.

---

### 2026-07-29 — Phase 2 Restructured and Logged

**Session goal**: Reorganize Phase 2 by establishing the `phase2/` directory and writing the corresponding reflection and documentation, using the reference conversation `ed561589-e027-4b6e-813b-6abef141d8eb`.

**Work done**:
- Created the `phase2/` directory and moved `Phase2_Data_Pipeline.ipynb` into it.
- Created `phase2/README.md` documenting the section-by-section breakdown of the pipeline, data choice justifications (SFT to ORPO, OpenHermes-2.5 replacing Alpaca), and critical observations like regex cleaning for conversational prefixes and filtering vs truncation.
- Updated root `README.md` to flag Phase 2 as `✅ Done` and Phase 3 as `🔜 Next`.
- Updated `project_knowledge.md` to reflect Phase 2 completion and restructure.

**State of project at this session**:
- Phase 1 (Tokenizer): Complete
- Phase 2 (Data Pipeline): Complete
- Phase 3 (QLoRA Setup): Next

---

### 2026-08-13 — Phase 3 Folder Created, Notebook Audited & Corrected

**Session goal**: Break down Phase 3, audit the existing `Phase3_Stage1_QLoRA_SFT.ipynb` against Tan's reference notebook, fix all issues, and set up the `phase3/` folder per project convention.

**Work done**:
- Read `Llama3.2-3B-final-fine-tune.ipynb` (Tan's notebook) end-to-end — extracted all LoRA and training hyperparameters (Q002 now ANSWERED)
- Audited our existing Phase 3 notebook against Tan's config — found 4 missing/wrong hyperparameters
- Resolved B001: confirmed our Phase 2 and Phase 3 both use `### Input:` / `### Response:` — format is consistent within our project
- Created `phase3/` directory
- Wrote corrected `phase3/Phase3_Stage1_QLoRA_SFT.ipynb` with all issues fixed and every hyperparameter deeply documented
- Wrote `phase3/README.md` with section-by-section breakdown, choices & whys, and reflections
- Updated root `README.md`: Phase 3 status → 🚧 In Progress
- Updated `project_knowledge.md`: Phase 3 section, hyperparameter registry, Q002/Q004/Q009 answered, B001 resolved

**Fixes applied to notebook**:
- Added `max_grad_norm = 0.3` (was missing; matches Tan)
- Added `group_by_length = True` (was missing; matches Tan)
- Added `eval_strategy = "steps"` with `eval_steps=200` (was missing)
- Corrected `weight_decay` 0.01 → 0.001 (matches Tan)
- Fixed sanity check decode logic to show only generated tokens
- Set `HF_USERNAME = "SpongeBob341"` (was placeholder)

**State of project at this session**:
- Phase 1 (Tokenizer): Complete
- Phase 2 (Data Pipeline): Complete (dataset not yet pushed to Hub — Q001 still open)
- Phase 3 (QLoRA Setup + Training): 🚧 In Progress — notebook ready, blocked on Q001 (Hub dataset repo name)
- Phase 4 (Training Loop): Merged into Phase 3
- Phase 5 (Evaluation): Not started
- Phase 6 (Server + Client): Not started

**Critical open item**: Q001 — the Phase 2 SFT dataset must be pushed to HuggingFace Hub and its repo name confirmed before Phase 3 notebook can be run. Everything else in Phase 3 is ready.

---

### 2026-08-14 — Phase 3 Live Test Run: Two Bugs Found & Fixed, Training Confirmed Working

**Session goal**: Execute the Phase 3 notebook on Colab T4, validating each cell in sequence.

**Cell-by-cell results**:
- Cell 1 (Install): ✅ All deps already present, Unsloth cloned from git (commit `08508fd`)
- Cell 2 (GPU check): ✅ Tesla T4, 14.56GB VRAM, bf16=False (fp16 path active)
- Cell 3 (HF login): ✅ Logged in as `SpongeBob387`
- Cell 4 (Config): ✅ All hyperparameters confirmed
- Cell 5 (Model load): ✅ 28 layers patched, 24.3M trainable params (r=16), pad token correct
- Cell 6 (Dataset): ✅ Q001 RESOLVED — `SpongeBob387/llama3.2-3b-sft-dataset` live. Train: 10,974 / Eval: 224
- Cell 7 (Training): ❌ → ❌ → ❌ → ✅ Three-round debugging process (B005 + B006)

**Bugs found and fixed**:

**B005 — Unsloth fused CE shape mismatch** (3 rounds to fix):
- Round 1: `TorchRuntimeError` — dynamo tracing of chunked CE failed. Fix attempt: `torch_compile=False`. Did not fully work (downgraded to plain ValueError).
- Round 2: `ValueError: batch_size (2048) vs (2052)` — chunk boundaries misaligned. Fix attempt: wrap original + disable compile. Did not fully work.
- Round 3: `TypeError: unsupported operand for *: Tensor and GradScaler` — `scaling` arg is a GradScaler object in fp16 mode, not a float. Fix: drop `* scaling`.
- Round 4: `ValueError: batch_size (2044) vs (2048)` — SFTTrainer appends EOS to labels (seq_len+1), after shift logits=511 tokens vs labels=512. Fix: `seq_len = min(logits.size(1), labels.size(1))` truncation. ✅ Training started.

**B006 — TRL SFTConfig migration** (discovered alongside B005):
- `TRL >= 0.10.0` deprecated `dataset_text_field`, `max_seq_length`, `packing` as direct `SFTTrainer` args. Must use `SFTConfig` instead of `TrainingArguments`.
- Fix: replaced `args=TrainingArguments(...)` with `args=SFTConfig(...)` and moved all SFT-specific args inside it.

**Key learning from B005 root cause**:
The 4-token-per-sample difference (2044 vs 2048 = 4×1) comes from:
1. SFTTrainer tokenizes each sample and appends EOS → labels have `seq_len + 1` tokens
2. Model forward pass uses `input_ids` of `seq_len` tokens → hidden_states of `seq_len`
3. After shift: logits = `seq_len - 1`, labels = `seq_len` → always 1 token off
4. The `min()` truncation removes the extra label token (the final EOS) — safe because EOS is `ignore_index=-100` in the loss anyway

**State of project at this session**:
- Phase 1 (Tokenizer): ✅ Complete
- Phase 2 (Data Pipeline): ✅ Complete (dataset live at `SpongeBob387/llama3.2-3b-sft-dataset`)
- Phase 3 (QLoRA Setup + Training): 🚧 In Progress — training running on T4 (test run, ~343 steps)
- Phase 4 (Training Loop): Merged into Phase 3
- Phase 5 (Evaluation): Not started
- Phase 6 (Server + Client): Not started

**Next**: Wait for training to complete, then run inference sanity check (Cell 7) and push adapter to Hub (Cell 8). Update this document with final train loss and runtime.

---

### 2026-08-14 — Phase 3 Complete: Training Run Finished, Adapter Pushed to Hub

**Session goal**: Complete Phase 3 training on T4, validate inference, push adapter.

**Final results**:
- Train Loss: `1.1717` | Best Eval Loss: `1.1367` (step 343)
- Runtime: **64.8 minutes** on T4 (faster than expected due to padding-free batching)
- Adapter pushed to: `SpongeBob387/llama3.2-3b-sft-stage1`

**Inference sanity check (3/3 passed)**:
1. Supervised vs Unsupervised — correct, detailed, cut off only by max_new_tokens=200
2. Math word problem — correct answer with GSM8K scratchpad format (`<<60*2.5=150>>`)
3. Capital of France — correct factual answer with relevant landmarks

**Additional finding — Padding-free batching**:
Unsloth auto-enabled padding-free batching via SFTConfig (Unsloth 2026.8 feature). Instead of padding all sequences to max_seq_length and computing on pad tokens, it concatenates samples into one long sequence with proper attention masking. This is why training was 15-20 min faster than the ~90 min estimate.

**Warnings logged (non-blocking)**:
- `max_new_tokens vs max_length`: Unsloth sets max_length=131072 internally; `max_new_tokens=200` correctly overrides it
- `clean_up_tokenization_spaces=True` on BPE tokenizer: Llama uses BPE, this setting is for WordPiece. Add `tokenizer.clean_up_tokenization_spaces = False` when loading tokenizer in future notebooks.

**State of project**:
- Phase 1 (Tokenizer): ✅ Complete
- Phase 2 (Data Pipeline): ✅ Complete
- Phase 3 (QLoRA SFT): ✅ Complete — adapter at `SpongeBob387/llama3.2-3b-sft-stage1`
- Phase 4 (Training Loop): Merged into Phase 3 ✅
- Phase 4.5 (ORPO Alignment): **Next** — merge Stage 1 adapter → run ORPOTrainer
- Phase 5 (Evaluation): Not started
- Phase 6 (Server + Client): Not started

---

> **MAINTENANCE RULE**: Every time a meaningful change is made — experiment run, hyperparameter confirmed, phase completed, bug found, design decision made — update the relevant section AND add a Journal entry. Do not let this document go stale. If you are the AI assistant reading this: **always read this file before starting any work on this project**. If you find a better plan or better information, update this document — do not leave it stale out of caution.
