# Phase 4.5 — Stage 2: ORPO Alignment

> **Goal:** Load the Stage 1 SFT adapter, inject fresh LoRA adapters, and run `ORPOTrainer` on our Phase 2 preference dataset (UltraFeedback Binarized, HH-RLHF Harmless, HaluEval) to produce a safety-aligned, hallucination-resistant final model.

---

## What This Phase Is About

Phase 3 taught the model *how to respond* — instruction format, structure, math reasoning, multi-hop QA. But knowing *how to respond* is not the same as knowing *what kind of response to give*. A model trained only on SFT can still produce harmful suggestions, confidently wrong facts, or low-quality verbose answers when better alternatives exist.

Phase 4.5 is the alignment layer. We take the Stage 1 SFT model and further train it with **ORPO** — a method that shows the model paired examples of `(chosen response, rejected response)` for the same prompt, and optimizes it to systematically prefer the better answer. This directly encodes human preferences into the model weights.

The two-stage structure (SFT → ORPO) is deliberate: SFT establishes the baseline skills that ORPO then refines. Running ORPO on the raw base model (without Stage 1) would fail because the model has no instruction-following behavior to align yet.

---

## Files

| File | Purpose |
|------|---------|
| [`Phase4_Stage2_ORPO_Alignment.ipynb`](./Phase4_Stage2_ORPO_Alignment.ipynb) | Full Stage 2 pipeline: load Stage 1 → inject LoRA → load ORPO dataset → ORPOTrainer → sanity check → push merged model. Run on Colab T4 GPU. |

**Run Instructions:**
1. Upload to Google Colab. Set runtime: **Runtime → Change runtime type → T4 GPU**.
2. Run Cell 1: installs Unsloth, TRL, datasets.
3. Run Cell 2: `notebook_login()` — authenticate with HuggingFace (write access required).
4. Run Cell 3: verify configuration — confirm `HF_USERNAME`, `STAGE1_ADAPTER`, `ORPO_DATASET`.
5. Run Cell 4: loads Stage 1 model + injects fresh Stage 2 LoRA adapters.
6. Run Cell 5: loads ORPO dataset and validates column format.
7. Run Cell 6: configures and runs `ORPOTrainer`. Training ~60-90 min on T4.
8. Run Cell 7: qualitative sanity check (3 prompts).
9. Run Cell 8: merges adapter into base model and pushes to HuggingFace Hub.

---

## What I Learned — Section by Section

### Section 1: Why ORPO Over Other Alignment Methods

The reference project (Tan) attempted DPO (Direct Preference Optimization) and the model broke post-training — responses became incoherent. DPO requires a frozen **reference model** and a KL-divergence penalty to prevent the policy from straying too far. Calibrating this penalty (`β_DPO`) is tricky: too low and the model degrades, too high and alignment is ineffective.

ORPO is strictly simpler. Its full loss is:

```
L_ORPO = L_CE(chosen) + β × L_OR(chosen, rejected)
```

- **`L_CE(chosen)`** is just standard language model cross-entropy on the chosen response. This is identical to what SFTTrainer runs on Alpaca data. It acts as a built-in safety net — the model can't forget how to generate coherent text because it's always being rewarded for producing good chosen responses.
- **`L_OR(chosen, rejected)`** is the odds-ratio penalty: it computes the log-odds of each response and penalizes cases where the model assigns similar log-odds to both. No reference model, no KL term, no separate reward model.

The `β` parameter (called `beta` in TRL) controls the relative weight. We use `β = 0.1` — conservative, prioritizing preserving Stage 1 instruction-following over aggressive alignment. This maps to: *"I want the model slightly more safety-aware, not dramatically different."*

### Section 2: The Merge-First Question (and Why We Skip It)

The naive approach to "Stage 2" would be:
1. Load Stage 1 adapter
2. `merge_and_unload()` to get a full merged model
3. Re-quantize in 4-bit
4. Apply new LoRA

The problem: step 3 is hard. `merge_and_unload()` produces a full-precision (bf16/fp16) model. Re-quantizing a 16-bit model back to 4-bit via BitsAndBytes requires reloading it with `from_pretrained(..., load_in_4bit=True)`, which means downloading it again or saving it to disk (6-7GB) first.

The simpler approach — and what we do — is:
- Load Stage 1 adapter repo directly via `FastLanguageModel.from_pretrained(STAGE1_ADAPTER)`
- Unsloth reads `adapter_config.json`, finds the base model name, loads base in 4-bit, then overlays the Stage 1 adapter weights
- Call `FastLanguageModel.get_peft_model()` which injects **new** trainable LoRA matrices while keeping everything else frozen

The Stage 1 behavior is preserved implicitly because those weights are part of the frozen "effective base" that the Stage 2 loss backpropagates through.

### Section 3: Two Forward Passes Per Step — the VRAM Budget Reality

SFTTrainer runs one forward pass per batch step: `(prompt + response)` → loss → gradient.

ORPOTrainer runs **two** forward passes per step:
1. `(prompt + chosen)` → `log P(chosen)` → L_CE + partial L_OR
2. `(prompt + rejected)` → `log P(rejected)` → complete L_OR

This doubles the peak activation memory per step vs. Stage 1 SFT. With `batch_size=2` and `seq_len=512`, each forward pass holds roughly `2 × 512 × 3072 × ~10 layers of hidden states`. On a T4 with 15GB, `batch_size=2` is the safe floor. Increasing to `batch_size=4` would likely OOM.

The gradient accumulation of 16 steps maintains our effective batch size of 32 without accumulating more than 2 samples of activations at a time.

### Section 4: Configuration — What `β`, LR, and `max_prompt_length` Actually Do

**`β = 0.1`**: The odds-ratio term is downweighted to 10% of the total loss. This means the SFT component (90%) dominates — Stage 1 instruction-following behavior is reinforced more strongly than alignment behavior is pushed. For our hardware and dataset size (~6k ORPO samples), this is the right trade-off. On a larger, higher-quality preference dataset, raising β to 0.2-0.3 would make sense.

**`learning_rate = 8e-5`**: Stage 1 used `2e-4`. We use 40% of that. The rule of thumb for Stage 2 fine-tuning: smaller LR = smaller gradient steps = less risk of overwriting Stage 1 weight patterns that encode instruction-following behavior. The project design specifies a range of `5e-5` to `1e-4` — `8e-5` sits comfortably in that range.

**`max_prompt_length = 256`**: ORPOTrainer splits each sample into `(prompt, response)` before tokenizing. Prompts exceeding `max_prompt_length` are truncated. Our Phase 2 ORPO prompts include the instruction and optional context — typically 80-200 tokens — so 256 gives comfortable headroom. The remaining 256 tokens go to the response.

### Section 5: Training Metrics — What to Watch

TRL's ORPOTrainer logs several sub-metrics beyond the total `loss`:

| Metric | What it Means | Healthy Trend |
|--------|--------------|---------------|
| `loss` | Total ORPO loss = SFT + β × OR | Decreasing |
| `sft_loss` | Cross-entropy on chosen | Decreasing |
| `odds_ratio_loss` | Alignment penalty | Decreasing, but volatile |
| `rewards/chosen` | Mean log-prob of chosen | Increasing |
| `rewards/rejected` | Mean log-prob of rejected | Decreasing or flat |
| `rewards/margins` | `chosen - rejected` | Increasing — key alignment indicator |

The most important single metric is **`rewards/margins`**. If it stays near 0 across all steps, the model is not learning to differentiate chosen from rejected — something is wrong (wrong dataset format, β too low, LR too low). If it shoots up immediately then plateaus, it may be overfitting.

### Section 6: The Save Strategy — Adapter vs. Merged

We push two artifacts:
- **Adapter only** (~80-160MB): Lightweight backup. Requires base model to load. Good for version control and iteration.
- **Merged 16-bit model** (~6.7GB): Standalone — base weights + Stage 1 adapter + Stage 2 adapter all folded together. Required for vLLM in Phase 5, which cannot dynamically load PEFT adapters in all versions.

`push_to_hub_merged(..., save_method="merged_16bit")` handles the merge, shard, and upload in one call via Unsloth. The 16-bit format is compatible with vLLM, standard `transformers`, and GGUF quantization tools.

---

## Choices & Whys

### Why `β = 0.1` and Not Higher?

The ORPO paper tested `β` values from 0.05 to 0.5 on LLaMA models. Higher β produced stronger alignment but degraded general instruction following — the odds-ratio term can dominate and cause the model to refuse or deflect on benign prompts. With only ~6k ORPO samples (small run), higher β also risks overfitting to the specific rejected examples. Starting conservative at 0.1 and observing `rewards/margins` is the right approach. If margins plateau near 0, we raise β. If the sanity check shows the model becoming overly cautious on Check 1, we lower β.

### Why Not Evaluate Strategy = "epoch"?

With ~6k training samples and effective batch size 32, one epoch is roughly 187 gradient steps. Evaluating every 100 steps gives ~2 checkpoints per epoch — enough to catch mid-training degradation. Epoch-level evaluation would give only 1 checkpoint and `load_best_model_at_end` would be meaningless.

### Why Keep `paged_adamw_32bit`?

Same reason as Stage 1: paging optimizer states to CPU RAM frees ~320MB of GPU memory that would otherwise compete with the doubled activation load from ORPO's two forward passes. On T4, every megabyte counts.

### Why Push Both Adapter and Merged?

The adapter is cheap to push (~2 minutes) and serves as version control — if the merged model is corrupted during upload, the adapter can be reloaded. The merged model is what Phase 5 (vLLM inference) and Phase 6 (FastAPI server) actually need. Pushing both is a one-time cost during training, not a recurring overhead.

---

## Reflections

The most important conceptual shift in this phase is recognizing that alignment is not "extra training" layered on top of SFT — it is a *different objective* that requires a *different data format*. SFT data says: "given this input, produce this output." Preference data says: "given this input, this response is better *than* that response." The model learns not just what to produce, but what to *avoid* producing.

ORPO makes this tractable without a reference model because the odds-ratio naturally encodes relative preference. The denominator `(1 - P(x))` of the log-odds acts as an implicit self-referencing baseline — the model compares its preference for chosen vs. rejected against its own current distribution, not a frozen snapshot of a previous model state.

The second key insight is how the two-stage design avoids the catastrophic forgetting problem that broke Tan's DPO experiment. By keeping `β` small and `LR` conservative, Stage 2 ORPO adds a thin preference-alignment layer on top of the Stage 1 foundation without erasing it. The `L_CE(chosen)` term is the explicit mechanism that preserves Stage 1 skills — every ORPO step also trains the model to produce high-quality chosen responses, reinforcing the SFT objective from Phase 3.

---

## What's Next → Phase 5: Inference & Evaluation

With the Stage 2 aligned model pushed to Hub, the next step is:
1. **Load** the merged model (`SpongeBob387/llama3.2-3b-orpo-stage2`) via **vLLM** with PagedAttention.
2. **Evaluate** on held-out benchmark samples from MMLU, CommonsenseQA, and TruthfulQA — datasets we explicitly excluded from training to prevent contamination.
3. **Compute** ROUGE-L and BERTScore across a diverse prompt set.
4. **Compare** quantitatively against a Stage 1-only baseline to measure the ORPO alignment delta.
