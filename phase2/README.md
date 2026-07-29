# Phase 2 — Data Pipeline (SFT & ORPO Dataset Curation)

> **Goal:** Build a robust, quality-centric, two-stage data preparation pipeline to support Stage 1 SFT (Skills) and Stage 2 ORPO (Alignment) while preventing benchmark contamination.

---

## What This Phase Is About

In this phase, we transition from examining individual tokens in isolation (Phase 1) to preparing the actual datasets that the model will learn from. Rather than training on a single monolithic dataset, we implement a **two-stage training architecture** requiring two distinct data formats:
1. **Track A (SFT Data):** A curated mixture of instruction-following, math, and multi-hop reasoning datasets formatted in a unified Alpaca-style structure to establish baseline capabilities.
2. **Track B (ORPO Data):** A curated mixture of preference pairs (`prompt`, `chosen`, `rejected`) focused on response quality, helpfulness, safety (harmlessness), and truthfulness (hallucination reduction).

Crucially, this pipeline runs entirely on CPU (Colab CPU or local CPU) and incorporates **eval-contamination safeguards** to ensure that benchmark datasets (MMLU, CommonsenseQA, TruthfulQA) are strictly excluded from all training data.

---

## Files

| File | Purpose |
|------|---------|
| [`Phase2_Data_Pipeline.ipynb`](./Phase2_Data_Pipeline.ipynb) | Jupyter notebook containing the full data loading, parsing, formatting, filtering, and hub-pushing pipeline. |

**Run Instructions:**
1. Open the notebook in Google Colab (CPU runtime is sufficient).
2. Install dependencies: `pip install transformers datasets huggingface_hub`.
3. Set `SMALL_RUN = True` for a quick validation run, or `False` for the production-grade run.
4. Input your Hugging Face credentials via `notebook_login()`.
5. Execute the cells sequentially to parse the datasets and upload the formatted SFT and ORPO splits to your HF Hub.

---

## What I Learned — Section by Section

### Section 1: Config and Setup
- **Quality > Quantity Strategy:** Training on massive, noisy datasets like Alpaca can degrade instruction adherence and introduce contradictions. Instead, we establish a schema using `SMALL_RUN` and `MAX_SAMPLES_SFT`/`MAX_SAMPLES_ORPO` limits to balance representation:
  - **SFT Max:** 3,000 (small validation) or 10,000 (full production) samples per dataset.
  - **ORPO Max:** 2,000 (small validation) or 7,500 (full production) samples per dataset.
- **Sequence Length Enforcer:** SFT and ORPO datasets have differing sequence lengths. To respect our hardware/memory constraint of `max_seq_length = 512`, we must filter samples proactively.

### Section 2: Shared Formatters
- **SFT Prompt Formatter (`text` column):** Converts inputs and completions to:
  ```text
  Instruction:
  {instruction_text}

  ### Input:
  {optional_context}

  ### Response:
  {response_text}<|end_of_text|>
  ```
  *Note: The `<|end_of_text|>` token is explicitly appended at the end of the response block to teach the model a clear stopping boundary.*
- **ORPO Pair Formatter (`prompt`, `chosen`, `rejected` columns):** ORPO aligns the model using relative comparison. It expects a dictionary:
  - `prompt`: `Instruction:\n{instruction}\n\n### Response:\n` (with optional context if `### Input:` is present)
  - `chosen`: `{chosen_text}<|end_of_text|>`
  - `rejected`: `{rejected_text}<|end_of_text|>`

### Section 3: Track A — SFT Dataset Curation
We ingest and format five core capability datasets:
1. **OpenHermes-2.5** (`teknium/OpenHermes-2.5`): Replaces Alpaca with GPT-4 tier high-quality instructions. We parse both its ShareGPT-style conversation lists and standard instruction/output shapes.
2. **Dolly-15k** (`databricks/databricks-dolly-15k`): Diverse, human-authored instructions covering brainstorming, summarizing, and editing.
3. **GSM8K** (`openai/gsm8k`): Grade-school math questions with step-by-step reasoning. We prefix the instruction with `"Solve the following math problem step by step."`
4. **MetaMathQA** (`meta-math/MetaMathQA`): Mathematically augmented queries ensuring multi-step arithmetic reasoning.
5. **HotpotQA** (`hotpot_qa`): High-quality multi-hop contextual QA. We reconstruct the passages in context (`title: context_sentence`) and inject it into the `### Input:` field.

### Section 4: Track B — ORPO Preference Curation
We ingest and format three preference datasets:
1. **UltraFeedback Binarized** (`HuggingFaceH4/ultrafeedback_binarized`): Focuses on response quality and helpfulness. We extract the terminal messages in the chosen/rejected paths.
2. **Anthropic HH-RLHF (Harmless)** (`Anthropic/hh-rlhf`): Aligns the model on safety. We use regular expressions to parse out the conversational prefixes (`Human:` and `Assistant:`) to strip out raw chat labels and isolate prompt vs response.
3. **HaluEval** (`pminervini/HaluEval`): Truthfulness and factual recall. Prepares correct answers as `chosen` and hallucinated answers as `rejected`, utilizing the facts as `### Input:` context.

### Section 5: Benchmark Contamination Safeguard
- A key pitfall in LLM SFT is data contamination. If evaluation sets are present in the training corpus, evaluation results are artificially inflated.
- We strictly exclude **MMLU** (`cais/mmlu`), **CommonsenseQA** (`commonsense_qa`), and **TruthfulQA** (`truthful_qa`) from both SFT and ORPO training pools. They are quarantined strictly for final model evaluation in Phase 5.

### Section 6: Filtering & Upload
- **Filter vs. Truncate:** Truncating text that exceeds `max_seq_length = 512` is highly damaging because it cuts off the model's `<|end_of_text|>` token and the actual completion, meaning the model trains on incomplete responses. We instead **discard** any samples whose token length exceeds 512.
- **Train/Test Splits:** Both formatted datasets are split into a `train` (98%) and `test` (2%) block and pushed to the Hugging Face Hub under private repositories.

---

## Choices & Whys

### Why SFT to ORPO (Two-Stage)?
Instead of a single-phase training or complex RLHF with separate reward models, we opted for:
1. **Stage 1 (QLoRA SFT):** Implants base skills, logical reasoning, and instructions.
2. **Stage 2 (QLoRA ORPO):** Directly optimizes the odds-ratio of chosen vs. rejected outputs. Because ORPO has an internal SFT loss component, it doesn't drift away from the base skills, provided the `chosen` responses are high-quality.

### Why replace Alpaca with OpenHermes-2.5?
Alpaca contains a lot of noisy, synthetic generation from older GPT-3 models. OpenHermes-2.5 represents clean, modern GPT-4 generations that teach cleaner instruction-following logic, aligning with our "Quality > Quantity" design choice.

### Why filter out long sequences instead of truncating them?
If you truncate a training sample to 512 tokens:
- You lose the completion response (the target output).
- The model trains on predicting an abrupt cutoff instead of the `<|end_of_text|>` end token.
Proactive filtering ensures the model only trains on complete, logical round-trips.

---

## Reflections

One of the most valuable realizations in this phase is how critical **data cleaning and regex parsing** are. Standard Hugging Face datasets are messy. Anthropic's HH-RLHF, for example, embeds dialog history inside a single string with `\n\nHuman:` and `\n\nAssistant:` prefixes. If we fed that raw text to the model, it would learn to generate those exact headers, clashing with our Alpaca format. Writing dedicated regex parsers for each dataset to map them onto our standard prompt structure ensures training consistency.

Additionally, this phase highlighted that **data-centric AI** is key. High-quality instruction sets (GSM8K, OpenHermes) combined with rigorous preference pairs (UltraFeedback, HaluEval) are vastly superior to simply dump-loading 88,000 raw, unvetted tokens.

---

## What's Next → Phase 3: QLoRA Setup

With SFT and ORPO datasets pushed to Hugging Face Hub, the next step is building the QLoRA configuration. We will define the quantization parameters (NF4, double quantization) and PEFT adapter targets (LoRA rank `r`, alpha, and target modules) to prepare the base model for Stage 1 training.
