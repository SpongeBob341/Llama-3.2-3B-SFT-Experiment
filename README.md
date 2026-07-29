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
| 3 | `phase3/` | QLoRA Setup — PEFT, BitsAndBytes, LoRA config |  Next |
| 4 | `phase4/` | Training Loop — SFTTrainer, hyperparameters, monitoring |  Planned |
| 5 | `phase5/` | Inference & Evaluation — vLLM, self-consistency, ROUGE |  Planned |
| 6 | `phase6/` | FastAPI Server & Client — deployment architecture | Planned |

### Current Plan (Phase 3)
**QLoRA Setup** — The next step is to configure the quantization and parameter-efficient fine-tuning layers:
- Set up **BitsAndBytes** configurations for 4-bit NormalFloat (`nf4`) quantization with double quantization.
- Configure **PEFT / LoRA** adapters, targeting correct projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.) in the attention mechanism.
- Verify memory allocation calculations to ensure the quantized model plus adapter weights fit within Google Colab's T4 GPU VRAM limits.


---



