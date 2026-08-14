# LLMAI: AI Assistant with Llama 3.2-3B

---

## 🗺️ Learning Journey & Plan

> This section documents the **ongoing plan** for building and deeply understanding this project from the ground up. It is only updated when there is a major shift in direction or scope.

> Each phase has its own folder (e.g., `phase1/`) containing the code and a detailed README with reflections, choices, and discoveries made during that phase.

### The Goal
I want to build a project that produces a working fine-tuned model. The goal now is to **understand it deeply** — from how the tokenizer works, to why each dataset was chosen, to why QLoRA was used over full fine-tuning, and what each training hyperparameter actually does.

### Phases

| Phase | Folder | Topic | Status |
|-------|--------|-------|--------|
| 1 | [`phase1/`](./phase1/) | Tokenizer Explorer — how the model sees text | ✅ Done |
| 2 | [`phase2/`](./phase2/) | Data Pipeline — loading & formatting SFT & ORPO datasets | ✅ Done |
| 3 | [`phase3/`](./phase3/) | QLoRA Setup + SFT Training Loop — Unsloth, LoRA, SFTConfig | ✅ Done |
| 4 | [`phase4/`](./phase4/) | Stage 2 ORPO Alignment — preference learning, safety, hallucination reduction | ✅ Done |
| 5 | [`phase5/`](./phase5/) | Inference & Evaluation — vLLM, self-consistency, ROUGE | 🚧 In Progress |
| 6 | `phase6/` | FastAPI Server & Client — deployment architecture | 🔜 Next |

### Current Plan (Phase 5 — Inference & Evaluation)
**Goal**: Evaluate the Stage 2 ORPO aligned model (`SpongeBob387/llama3.2-3b-orpo-stage2`) using high-throughput vLLM serving, self-consistency sampling, and benchmark metrics.
- Execute [`phase5/Phase5_Inference_and_Evaluation.ipynb`](./phase5/Phase5_Inference_and_Evaluation.ipynb) on Colab T4/A100 GPU.
- Evaluate on un-contaminated held-out benchmarks (MMLU, CommonsenseQA, TruthfulQA, GSM8K test split).
- Benchmark generation performance: ROUGE-L structural similarity and BERTScore semantic quality.
- Measure inference throughput (tokens/sec) comparing vLLM PagedAttention continuous batching vs standard HuggingFace pipeline.


---



