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

## Decision Conflicts & Strategy Comparisons

Building this data pipeline involved resolving several critical strategy conflicts, choosing between dataset scaling, alignment algorithms, and formatting constraints:

### 1. Alignment Algorithm: DPO vs. ORPO vs. RLHF
* **Conflict:** We needed a parameter-efficient alignment method to teach safety and truthfulness.
  * *RLHF (Reinforcement Learning from Human Feedback)* requires training a separate reward model and running PPO, which is too resource-heavy and complex for a 3B model under resource constraints.
  * *DPO (Direct Preference Optimization)* is highly popular but notoriously unstable. DPO training caused severe regression, breaking the model's ability to hold basic interactions.
* **Resolution:** We chose **ORPO (Odds-Ratio Preference Optimization)**. ORPO merges SFT and preference alignment into a single loss function. By penalizing the odds ratio of rejected responses relative to chosen ones directly during training, ORPO eliminates the need for a reference model or a reward model. Combined with a conservative learning rate and ultra-high-quality `chosen` preference samples, ORPO prevents the policy degradation seen in DPO.

### 2. Dataset Philosophy: Scale vs. Cleanliness (88k vs. ~60k Samples)
* **Conflict:** The reference project trained on 88,031 samples. To reproduce or beat this, should we maximize sample volume or focus on quality?
  * *Volume-centric* approaches include noisier, older synthetic datasets like Alpaca, which can introduce contradictions and lower-tier GPT-3 habits.
  * *Curation-centric* approaches restrict sample sizes but leverage cleaner, GPT-4 level instructions.
* **Resolution:** Locked in **Quality-First Curation** (~35k-45k SFT samples, ~20k ORPO preference pairs). We completely replaced Alpaca with **OpenHermes-2.5** (GPT-4 tier instruction data). This guarantees the model learns high-fidelity reasoning patterns, proving that a smaller, cleaner dataset outperforms a larger, noisier one.

### 3. Evaluation Safeguard: Contamination vs. Knowledge Coverage
* **Conflict:** To teach the model general knowledge, common sense, and truthfulness, it is tempting to train on datasets like MMLU, CommonsenseQA, and TruthfulQA. However, these are standard evaluation benchmarks.
* **Resolution:** We chose **Strict Benchmark Quarantine**. Including these datasets in the training split causes data leakage, making downstream evaluation metrics useless. We explicitly excluded these datasets from all SFT and ORPO pipelines, reserving them purely as clean, unseen evaluation benchmarks for Phase 5.

### 4. Sequence Length Limit: Truncation vs. Discard/Filtering
* **Conflict:** To respect the hard memory constraint of `max_seq_length = 512` (necessary for T4 VRAM limits), how should we handle sequences exceeding this length?
  * *Truncating* keeps the sample count high but chops off the completion and the trailing `<|end_of_text|>` token.
  * *Filtering (Discarding)* preserves completion formatting but reduces total dataset size.
* **Resolution:** We chose **Filtering (Discarding)**. Truncation is highly destructive; if the model trains on truncated completions, it learns to generate cut-off responses and never learns the stop token, causing infinite rambling at inference. Correct formatting behavior is prioritized over raw sample count.

### 5. Dialog Format: Built-in Chat Template vs. Alpaca Schema
* **Conflict:** Instruct models use built-in Chat Templates (like Llama-3-Instruct's Jinja template). Since we are fine-tuning the base `Llama-3.2-3B` model (which lacks pre-training on a specific instruct format), using standard chat templates might introduce mismatched pre-training signals.
* **Resolution:** We locked in the **Alpaca Prompt Schema** (using `Instruction:`, `### Input:`, and `### Response:` blocks). This is simple, consistent, maps perfectly to both instruction-only and context-inclusive samples, and ensures 1-to-1 parity between our training formatting and the downstream inference server.

---


## Reflections

One of the most valuable realizations in this phase is how critical **data cleaning and regex parsing** are. Standard Hugging Face datasets are messy. Anthropic's HH-RLHF, for example, embeds dialog history inside a single string with `\n\nHuman:` and `\n\nAssistant:` prefixes. If we fed that raw text to the model, it would learn to generate those exact headers, clashing with our Alpaca format. Writing dedicated regex parsers for each dataset to map them onto our standard prompt structure ensures training consistency.

Additionally, this phase highlighted that **data-centric AI** is key. High-quality instruction sets (GSM8K, OpenHermes) combined with rigorous preference pairs (UltraFeedback, HaluEval) are vastly superior to simply dump-loading 88,000 raw, unvetted tokens.

---

## What's Next → Phase 3: QLoRA Setup

With SFT and ORPO datasets pushed to Hugging Face Hub, the next step is building the QLoRA configuration. We will define the quantization parameters (NF4, double quantization) and PEFT adapter targets (LoRA rank `r`, alpha, and target modules) to prepare the base model for Stage 1 training.
