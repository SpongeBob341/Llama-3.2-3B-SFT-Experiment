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
| 1 | [`phase1/`](./phase1/) | Tokenizer Explorer — how the model sees text |  Done |
| 2 | [`phase2/`](./phase2/) | Data Pipeline — loading & formatting SFT & ORPO datasets |  Done |
| 3 | [`phase3/`](./phase3/) | QLoRA Setup + SFT Training Loop — Unsloth, LoRA, SFTConfig | ✅ Done |
| 4 | `phase4/` | Training Loop — SFTTrainer, hyperparameters, monitoring |  Planned |
| 5 | `phase5/` | Inference & Evaluation — vLLM, self-consistency, ROUGE |  Planned |
| 6 | `phase6/` | FastAPI Server & Client — deployment architecture | Planned |

### Current Plan (Phase 4.5 — Stage 2 ORPO Alignment)
**Next step**: Before running ORPO, we must merge the Stage 1 LoRA adapter into the base model weights.
- Load Stage 1 adapter from `SpongeBob387/llama3.2-3b-sft-stage1`, merge into base `Llama-3.2-3B` using `model.merge_and_unload()`.
- Run `ORPOTrainer` on the Phase 2 ORPO dataset (UltraFeedback, HH-RLHF Harmless, HaluEval) for preference alignment.
- Push the Stage 2 aligned model to HuggingFace Hub — this is the final model that goes into the Phase 5 inference server.


---



