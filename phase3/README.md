# Phase 3 & 4 — QLoRA Setup + SFT Training Loop

> **Goal:** Configure 4-bit NF4 quantization and LoRA adapters via Unsloth, load our Phase 2 SFT dataset from HuggingFace Hub, and run `SFTTrainer` to produce a Stage 1 fine-tuned model checkpoint.

---

## What This Phase Is About

This is the phase where the model actually learns. Phases 1 and 2 were preparation — understanding tokenization, curating datasets, and building the data pipeline. Phase 3 is where all of that comes together: we take the raw `Llama-3.2-3B` base model, wrap it in QLoRA (4-bit NF4 quantization + LoRA adapters), feed it our curated SFT dataset, and train it to behave as an instruction-following assistant.

The decision to combine Phase 3 (QLoRA Setup) and Phase 4 (Training Loop) into a single notebook reflects that they are not truly separable: you cannot set up QLoRA without loading a model, and you cannot train without configuring QLoRA. The notebook is structured as a linear sequence from setup to push — run top to bottom, once, on Colab GPU.

---

## Files

| File | Purpose |
|------|---------|
| [`Phase3_Stage1_QLoRA_SFT.ipynb`](./Phase3_Stage1_QLoRA_SFT.ipynb) | Full training pipeline: Unsloth setup → QLoRA model load → LoRA adapter injection → dataset load → SFTTrainer → sanity check → Hub push. Run on Colab GPU (T4 minimum). |

**Run Instructions:**
1. Upload to Google Colab. Set runtime to **GPU** (Runtime → Change runtime type → T4 or A100).
2. Run Cell 1: install Unsloth and dependencies.
3. Run Cell 2: verify GPU and VRAM.
4. Run Cell 3: `notebook_login()` — authenticate with HuggingFace.
5. In Cell 4 (Configuration): set `HF_USERNAME` to your username and confirm `SFT_DATASET_REPO` matches the Phase 2 Hub repo.
6. Run all remaining cells in sequence. Training takes ~1-2 hours on T4 for a ~35k sample dataset.

---

## What I Learned — Section by Section

### Section 1: Setup — Why Unsloth?

The reference project (Tan's notebook) loads the model with the raw HuggingFace stack: `AutoModelForCausalLM.from_pretrained()` with a manually constructed `BitsAndBytesConfig`, then calls `prepare_model_for_kbit_training()` and injects LoRA via `get_peft_model()`. This works but is slow and memory-hungry.

We use **Unsloth** instead. Unsloth is not an alternative framework — it wraps the same HuggingFace model, uses the same PEFT adapters, and produces the same artifact. What it changes is the *implementation* of the attention and cross-entropy backward passes: Unsloth replaces them with hand-written Triton/CUDA kernels that are 2-5x faster and use 60% less VRAM. The model it produces is binary-identical to a vanilla QLoRA model.

The practical consequence: on a T4 (15GB), Unsloth makes training feasible where the raw stack often runs out of memory or requires painful micro-batch sizes.

### Section 2: Configuration — What Every Hyperparameter Does

**`lora_r = 16`**: The rank of the LoRA adapter matrices. Each target layer injects two matrices (A: d × r, B: r × d) whose product approximates the weight update. Higher rank = more expressiveness but more VRAM. Tan's reference used r=64 on an A100 80GB — feasible there but would exhaust a T4. We use r=16, which still produces ~40M trainable parameters across all 7 target layers.

**`lora_alpha = 16`**: The scaling factor. The LoRA update is `(alpha/r) × B×A`. With alpha=r=16, the scaling factor is 1.0, which means the adapter output is applied at full weight. Common variants: alpha=2r doubles the effective learning rate of the adapters.

**`lora_dropout = 0`**: Unsloth's custom kernels are written for the common case (no dropout). Non-zero dropout triggers an internal fallback to the vanilla HF implementation, sacrificing the CUDA speedup. Zero dropout is the right choice when Unsloth is being used.

**`max_grad_norm = 0.3`**: Gradient clipping. If the L2 norm of all gradients exceeds 0.3, they are rescaled proportionally so their norm equals 0.3 exactly. This prevents the rare but catastrophic "exploding gradient" events where a single bad batch causes weight updates so large they destroy all prior learning. This was in Tan's config but was missing from the original Phase 3 notebook.

**`group_by_length = True`**: Before building each batch, the trainer groups training samples by similar token length. This minimizes the amount of padding needed to make sequences the same length within a batch. On a 35k sample dataset, this typically gives 5-15% faster training at zero quality cost. Also absent from the original notebook.

### Section 3: Loading QLoRA — What Actually Happens at the CUDA Level

When `FastLanguageModel.from_pretrained()` is called with `load_in_4bit=True`, the following happens:

1. The full-precision checkpoint is loaded shard by shard.
2. Each weight tensor is **quantized from fp32/bf16 to NF4** (Normal Float 4): the weights are mapped to the 16 values in the NF4 codebook (which are spaced to minimize quantization error for normally distributed values — hence "normal float").
3. A **double quantization** layer is applied on top: the quantization constants themselves are quantized from fp32 to 8-bit, saving an additional ~0.4 bits per parameter.
4. The quantized weights are stored in GPU VRAM. The ~3.3B parameter model goes from ~6GB → ~1.8GB GPU footprint.

The LoRA adapters (`A` and `B` matrices for each target layer) are then initialized (A~random normal, B=0 so the initial adapter output is zero) and kept in bf16/fp16. Only these ~40M parameters are updated during training — the base model stays frozen at 4-bit.

### Section 4: Dataset Loading

The Phase 2 pipeline produced a dataset in Alpaca format with a single `text` column. `SFTTrainer` is told to use `dataset_text_field="text"` — it will tokenize these pre-formatted strings internally, pad them to `MAX_SEQ_LENGTH=512`, and feed them to the model. No additional formatting happens at training time.

### Section 5: The Training Loop — paged_adamw_32bit

The optimizer `paged_adamw_32bit` is a memory-efficient variant of AdamW designed specifically for QLoRA. Standard AdamW stores two optimizer states per parameter (first and second moment estimates), both in fp32. For our ~40M trainable parameters, that's 40M × 2 × 4 bytes = **320MB of optimizer state**. On a T4, this is significant.

`paged_adamw_32bit` stores these optimizer states in CPU RAM by default and "pages" them (copies them) to GPU only when a gradient update is needed. The result is dramatically reduced GPU memory usage with a modest CPU→GPU transfer overhead that typically doesn't bottleneck training.

### Section 6: Sanity Check — Why Decode Only New Tokens

After training, the sanity check cell decodes only the *generated* tokens, not the prompt. The original Phase 3 notebook used `skip_special_tokens=True` on the full output, which truncates correctly but includes the prompt text in the printout. The corrected version slices `outputs[0][inputs['input_ids'].shape[1]:]` to return only what the model generated, making the output easier to read.

---

## Choices & Whys

### Why Not Use the Raw HuggingFace Stack (Like Tan)?

Tan's notebook manually constructs `BitsAndBytesConfig`, calls `prepare_model_for_kbit_training()`, and applies `get_peft_model()` separately. Our notebook uses `FastLanguageModel.from_pretrained()` which does all of this in one call with optimized kernels. The choice is purely pragmatic: same model, faster and leaner on Colab-tier hardware.

### Why r=16 Instead of r=64?

r=64 requires ~4× the LoRA parameter count vs. r=16. On a T4, this moves the adapter matrices from ~160MB to ~640MB — significant when total VRAM is only 15GB. r=16 is the standard recommendation for T4 SFT runs and still achieves meaningful fine-tuning. If training on an A100, raising to r=64 would match Tan's exact config.

### Why `packing=False`?

Dataset packing concatenates multiple short samples into one 512-token window, improving GPU utilization. However, our Phase 2 pipeline used Alpaca-format samples that vary significantly in length, and the loss can bleed across samples at boundaries if attention masking is not carefully implemented. `packing=False` is the safer choice and what Tan used.

### Why Keep eval_strategy="steps" Instead of "epoch"?

With 1 epoch of training (~35k samples / 32 effective batch size ≈ ~1,100 gradient steps on a small run), evaluating at epoch end gives only one validation loss measurement. Evaluating every 200 steps gives 5-6 checkpoints during training — enough to detect if the model is degrading mid-epoch and to `load_best_model_at_end`.

---

## Issues Found & Fixed in the Original Notebook

The `Phase3_Stage1_QLoRA_SFT.ipynb` file at the root of the repo had these problems:

| Issue | Original | Fixed |
|-------|----------|-------|
| `max_grad_norm` missing | Not set | Added: `0.3` |
| `group_by_length` missing | Not set | Added: `True` |
| `eval_strategy` missing | Not set | Added: `"steps"` with `eval_steps=200` |
| `weight_decay` | `0.01` | Corrected to `0.001` (matches Tan) |
| `HF_USERNAME` | `"your-username"` | Set to `"SpongeBob341"` |
| Decode logic in sanity check | Decoded full output including prompt | Sliced to new tokens only |
| Comments | Minimal | Comprehensive — explains every hyperparameter |

---

## Reflections

The most important realization in this phase is how much of the "magic" of QLoRA happens before a single training step. By the time `trainer.train()` is called, the model is already in an unusual state: most of its parameters are frozen at 4-bit precision, and only a small set of injected matrices (about 1/80th of the parameter count) will be updated. The entire training signal — the gradient — flows from the loss, backward through the frozen dequantized weights, and accumulates only on those adapter matrices.

This means QLoRA has a natural limitation: the adapter can only learn to shift the model's behavior within the subspace spanned by the LoRA rank. Choosing `r=16` means each target layer's adapter can represent at most 16 directions of weight update, no matter how long training runs. This is why rank matters and why Tan chose r=64 when he had the VRAM for it.

The second key insight is that **Unsloth is not optional for T4 training** — it is practically necessary. Without it, the naive QLoRA stack frequently OOMs during the first few batches on a T4, especially with gradient checkpointing enabled. Unsloth's memory reduction is what makes this project runnable on free Colab hardware.

---

## What's Next → Phase 4.5: Stage 2 ORPO Alignment

With the Stage 1 SFT adapter trained and pushed, the next step is:
1. **Merge** the Stage 1 adapter into the base model weights (producing a standalone merged model).
2. **Load** the merged model and apply ORPO training using `ORPOTrainer` on our Phase 2 ORPO dataset (UltraFeedback, HH-RLHF Harmless, HaluEval).
3. **Save and push** the Stage 2 aligned model — this is the final model that goes into the inference server (Phase 5).
