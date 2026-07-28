# LLMA-O: AI Assistant with Llama 3.2-3B

---

## 🗺️ Learning Journey & Plan

> This section documents the **ongoing plan** for building and deeply understanding this project from the ground up. It is only updated when there is a major shift in direction or scope.

> Each phase has its own folder (e.g., `phase1/`) containing the code and a detailed README with reflections, choices, and discoveries made during that phase.

### The Goal
I want to build a project that produces a working fine-tuned model, without cutting corners. The goal now is to **understand it deeply** — from how the tokenizer works, to why each dataset was chosen, to why QLoRA was used over full fine-tuning, and what each training hyperparameter actually does.

### Phases

| Phase | Folder | Topic | Status |
|-------|--------|-------|--------|
| 1 | [`phase1/`](./phase1/) | Tokenizer Explorer — how the model sees text |  Done |
| 2 | `phase2/` | Data Pipeline — loading & formatting all 10 datasets | Next |
| 3 | `phase3/` | QLoRA Setup — PEFT, BitsAndBytes, LoRA config |  Planned |
| 4 | `phase4/` | Training Loop — SFTTrainer, hyperparameters, monitoring |  Planned |
| 5 | `phase5/` | Inference & Evaluation — vLLM, self-consistency, ROUGE |  Planned |
| 6 | `phase6/` | FastAPI Server & Client — deployment architecture | Planned |

### Current Plan (Phase 2)
**Data Pipeline** — The next step is to understand and reproduce the full data loading and formatting pipeline:
- Load each of the 10 datasets from HuggingFace (`alpaca`, `dolly`, `hh-rlhf`, `gsm8k`, `MetaMathQA`, `hotpot_qa`, `mmlu`, `commonsense_qa`, `trivia_qa`, `TruthfulQA`)
- Write a separate formatting function for each dataset that converts raw samples into the standardized prompt format
- Cap each dataset at 10,000 samples, shuffle, and combine into a single unified dataset
- Tokenize a sample batch and verify token lengths fit within `max_seq_length = 512`

---



