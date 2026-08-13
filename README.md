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
| 3 | [`phase3/`](./phase3/) | QLoRA Setup + SFT Training Loop — Unsloth, LoRA, SFTTrainer | 🚧 In Progress |
| 4 | `phase4/` | Training Loop — SFTTrainer, hyperparameters, monitoring |  Planned |
| 5 | `phase5/` | Inference & Evaluation — vLLM, self-consistency, ROUGE |  Planned |
| 6 | `phase6/` | FastAPI Server & Client — deployment architecture | Planned |

### Current Plan (Phase 3 + 4 Combined)
**QLoRA Setup + SFT Training Loop** — We use Unsloth for a 2-5x faster, memory-efficient implementation:
- Load `Llama-3.2-3B` in 4-bit NF4 via Unsloth's `FastLanguageModel`, injecting LoRA into all 7 projection layers (`q, k, v, o, gate, up, down`).
- Pull the Phase 2 SFT dataset from HuggingFace Hub and train with `SFTTrainer` (cosine LR, paged AdamW, effective batch size 32 via gradient accumulation).
- Run qualitative sanity checks post-training, then push the LoRA adapter to HuggingFace Hub for use in Phase 4.5 (ORPO alignment).


---



