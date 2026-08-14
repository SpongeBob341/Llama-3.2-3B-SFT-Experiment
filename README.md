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
| 4 | [`phase4/`](./phase4/) | Stage 2 ORPO Alignment — preference learning, safety, hallucination reduction | 🔜 Next |
| 5 | `phase5/` | Inference & Evaluation — vLLM, self-consistency, ROUGE | ⬜ Planned |
| 6 | `phase6/` | FastAPI Server & Client — deployment architecture | ⬜ Planned |

### Current Plan (Phase 4.5 — Stage 2 ORPO Alignment)
**Goal**: Align the Stage 1 SFT model on human preference data to produce the final deployment-ready model.
- Load Stage 1 adapter (`SpongeBob387/llama3.2-3b-sft-stage1`) via Unsloth, inject fresh LoRA adapters for ORPO training.
- Run `ORPOTrainer` (β=0.1, LR=8e-5) on Phase 2 ORPO dataset (`SpongeBob387/llama3.2-3b-orpo-dataset`) — UltraFeedback, HH-RLHF Harmless, HaluEval preference pairs.
- Merge Stage 2 adapter into full 16-bit model and push to `SpongeBob387/llama3.2-3b-orpo-stage2` — this is the final model that goes into Phase 5 (vLLM inference) and Phase 6 (FastAPI server).


---



