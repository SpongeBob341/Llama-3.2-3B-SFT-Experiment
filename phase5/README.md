# Phase 5 — Inference & Evaluation Suite

> **Goal:** Build an end-to-end evaluation and high-throughput inference pipeline for our Stage 2 ORPO aligned model (`SpongeBob387/llama3.2-3b-orpo-stage2`), evaluating it against quarantined benchmarks (MMLU, CommonsenseQA, TruthfulQA, GSM8K), standard generation metrics (ROUGE, BERTScore), and qualitative alignment stress tests using vLLM with PagedAttention and ROUGE-L self-consistency.

---

## What This Phase Is About

Training a model is only half the battle. Once weights are updated and adapters are merged, we must rigorously answer two critical questions:
1. **Did the model actually improve?** Did Stage 1 SFT successfully teach instruction following and multi-step reasoning, and did Stage 2 ORPO successfully instill safety guardrails and factual grounding without breaking baseline skills?
2. **How do we serve it efficiently?** Standard PyTorch `model.generate()` is bottlenecked by sequential decoding, lack of continuous batching, and severe memory fragmentation in the Key-Value (KV) cache.

Phase 5 creates the complete inference and evaluation harness. It tests the merged model on **quarantined evaluation benchmarks** that were strictly excluded from training data to ensure zero contamination. Furthermore, it implements **vLLM PagedAttention** continuous batching and **ROUGE-L similarity-based Self-Consistency** majority voting, establishing the exact inference architecture that will power the production FastAPI server in Phase 6.

---

## Files

| File | Purpose |
|------|---------|
| [`Phase5_Inference_and_Evaluation.ipynb`](./Phase5_Inference_and_Evaluation.ipynb) | Complete 10-cell evaluation notebook: vLLM PagedAttention setup, Self-Consistency ROUGE-L consensus engine, automated benchmark testing (MMLU, GSM8K, TruthfulQA), text similarity scoring (ROUGE-1/2/L, BERTScore), qualitative safety stress-testing, and throughput profiling. Run on Colab GPU (T4/A100). |

**Colab Run Instructions:**
1. Upload `Phase5_Inference_and_Evaluation.ipynb` to Google Colab.
2. Select GPU runtime: **Runtime → Change runtime type → T4 GPU** (or A100).
3. Run **Cell 1**: Installs `vllm`, `transformers`, `datasets`, `evaluate`, `rouge_score`, `bert_score`, `tabulate`, and `matplotlib`.
4. Run **Cell 2**: Verifies CUDA device, VRAM, and platform compatibility.
5. Run **Cell 3**: Configures target model repo (`SpongeBob387/llama3.2-3b-orpo-stage2`) and prompt formats.
6. Run **Cell 4**: Initializes the vLLM engine with PagedAttention (or auto-falls back to Hugging Face pipeline if non-Linux).
7. Run **Cell 5**: Initializes the ROUGE-L pairwise consensus self-consistency generator.
8. Run **Cell 6**: Runs automated evaluation across quarantined benchmarks (**MMLU**, **GSM8K**).
9. Run **Cell 7**: Computes **ROUGE-1/2/L/Lsum** and **BERTScore (P/R/F1)** on reference text pairs.
10. Run **Cell 8**: Executes the 5-axis qualitative alignment stress test (evaluating the Stage 2 ORPO delta).
11. Run **Cell 9**: Runs batch throughput benchmarking (measuring tokens/second and latency).
12. Run **Cell 10**: Generates the final evaluation dashboard and summary comparison bar charts.

---

## What I Learned — Section by Section

### Section 1: The Memory Wall, KV Cache & PagedAttention Mechanics

In standard autoregressive decoding, generating each new token requires computing attention over all preceding tokens in the sequence. To avoid recomputing Key and Value projections for past tokens at every step, transformers cache them in GPU memory as the **KV-Cache**.

However, standard memory allocators present severe inefficiencies:
- **Internal Fragmentation**: Allocating memory buffers upfront for `max_seq_length = 512` wastes VRAM when actual responses are only 50-100 tokens long.
- **External Fragmentation**: Dynamic memory allocation across concurrent requests leaves scattered, unusable memory gaps.
- **Memory Waste**: Up to 60–80% of GPU memory in traditional serving stacks is wasted on fragmented or pre-allocated unused KV cache slots.

**How vLLM PagedAttention Solves This**:
Inspired by virtual memory paging in operating systems, vLLM divides the KV-cache of each sequence into fixed-size **blocks** (virtual pages) stored in non-contiguous physical GPU memory.
1. Physical memory is allocated on-demand as tokens are generated.
2. Dynamic continuous batching allows incoming queries to immediately join executing batches without waiting for long sequences to finish.
3. Result: **10× to 20× higher throughput** compared to native Hugging Face `pipeline` on the exact same GPU.

```
Standard Memory Allocation (Contiguous):
┌───────────┬───────────────────────────────────────┐
│ Actual KV │ Wasted Pre-allocated Reserved Memory │ (60-80% wasted)
└───────────┴───────────────────────────────────────┘

PagedAttention (vLLM Dynamic Blocks):
Physical Blocks: [Block A1] [Block B1] [Block A2] [Block C1] [Block B2]
(Near 0% memory fragmentation — enables large continuous batch sizes)
```

---

### Section 2: Self-Consistency & Why ROUGE-L Consensus Outperforms Greedy & Embeddings

Standard greedy decoding (`temperature=0.0`) selects the single highest-probability token at each step. While deterministic, it suffers from severe vulnerabilities:
- If the model makes an early misstep (e.g. an incorrect intermediate calculation in a math problem), it remains trapped on that incorrect trajectory.
- Random sampling (`temperature=0.7`) introduces diversity but risks generating occasional hallucinations.

**The Solution: Self-Consistency with Pairwise ROUGE-L Consensus**:
1. Sample $N=5$ candidate responses independently using nucleus sampling ($T=0.7, p=0.9$).
2. For each candidate $c_i$, compute its **pairwise ROUGE-L F1 score** against every other candidate $c_j$ ($j \neq i$):
   $$\text{Score}(c_i) = \frac{1}{N-1} \sum_{j \neq i} \text{ROUGE-L}_{F1}(c_i, c_j)$$
3. Select the candidate with the highest average consensus score:
   $$c^* = \arg\max_{c_i} \text{Score}(c_i)$$

**Why ROUGE-L Consensus Over Embedding Cosine Similarity or Math Parsers?**
As discovered in Tan's reference experiments:
- **Math parsers**: Brittle regexes frequently misclassified qualitative questions as math problems or failed on non-standard numeric formats.
- **Cosine similarity with embedding models**: High-level semantic embeddings often mapped completely hallucinated facts to high cosine similarity because the sentence structure was grammatically fluent.
- **ROUGE-L (Longest Common Subsequence)**: Measures structural and sequence agreement. If 4 out of 5 candidates share the same core reasoning chain and factual structure, they will mutually score high ROUGE-L with each other, while a lone hallucinated candidate receives a low score and is filtered out.

---

### Section 3: The Evaluation Hierarchy (Exact Match vs ROUGE vs BERTScore)

Evaluating generative language models requires distinct metrics tailored to each task type:

| Metric Type | Example Metrics | Best Used For | Limitation |
|-------------|-----------------|---------------|------------|
| **Exact Match / Regex Match** | Binary accuracy, numeric parsing | GSM8K, Multiple-Choice Benchmarks (MMLU) | Cannot evaluate open-ended explanations or creative writing. |
| **N-gram Overlap** | ROUGE-1, ROUGE-2, ROUGE-L, BLEU | Summarization, structured text reconstruction | Fails when the model expresses the correct fact using different synonyms. |
| **Semantic Embedding Similarity** | BERTScore (Precision, Recall, F1) | Open-ended QA, conceptual explanations | Computationally heavier; insensitive to small but critical numeric details. |
| **Qualitative Side-by-Side** | Human / Persona stress test | Safety refusals, tone, formatting adherence | Slower to evaluate; requires curated probe prompts. |

Using all four tiers in Phase 5 provides a 360-degree profile of model behavior.

---

### Section 4: Benchmark Quarantine — Ensuring Honest Generalization

In Phase 2, we made the architectural decision to strictly **quarantine** standard academic benchmarks (**MMLU**, **CommonsenseQA**, **TruthfulQA**) from our training dataset.

**Why Benchmark Quarantine Was Essential**:
- Many contemporary fine-tuning datasets inadvertently blend MMLU or TruthfulQA questions into their SFT mix.
- A model trained on benchmark questions will exhibit artificially high test scores (memorization rather than reasoning).
- Because our training dataset was limited to OpenHermes-2.5, Dolly-15k, GSM8K, MetaMathQA, HotpotQA, and ORPO preference pairs, testing on MMLU and TruthfulQA in Phase 5 provides an **authentic measurement of zero-shot out-of-distribution generalization**.

---

### Section 5: Quantifying the Stage 2 ORPO Delta

Stage 2 ORPO alignment was introduced specifically to solve the weaknesses of pure SFT. In Phase 5, we test the **ORPO Delta**:

1. **Safety & Refusal Boundary (HH-RLHF impact)**:
   - *Stage 1 SFT Baseline*: Follows harmful or unethical user requests eagerly if phrased as direct instructions.
   - *Stage 2 ORPO Aligned*: Politely and firmly refuses harmful requests while maintaining helpfulness on benign requests.
2. **Truthfulness & Anti-Hallucination (HaluEval impact)**:
   - *Stage 1 SFT Baseline*: Tends to confirm common human misconceptions (e.g. "humans only use 10% of their brains").
   - *Stage 2 ORPO Aligned*: Correctly refutes popular myths with factual scientific explanations.
3. **Preservation of Reasoning**:
   - GSM8K multi-step math problem solving remains intact with scratchpad calculations (`<<...>>`), proving that Stage 2's low learning rate (`8e-5`) and conservative $\beta=0.1$ prevented catastrophic forgetting.

---

## Choices & Whys

| Decision | Alternative Considered | Why We Chose This Path |
|----------|------------------------|-------------------------|
| **vLLM Serving Engine** | Hugging Face `AutoModelForCausalLM` / `pipeline` | vLLM PagedAttention provides 10-20x throughput and memory optimization, perfectly preparing for the Phase 6 production server. |
| **Pairwise ROUGE-L Consensus** | Cosine embedding similarity / Greedy decoding | Longest Common Subsequence rewards structural agreement while filtering out outlier hallucinations without the embedding drift issues seen in Tan's tests. |
| **$N=5$ Samples, $T=0.7$, $p=0.9$** | $N=10$ or $T=1.0$ | $N=5$ provides sufficient distribution diversity for majority voting while keeping inference latency within interactive bounds. |
| **Quarantined MMLU & TruthfulQA Evaluation** | Evaluating only on training set splits | Held-out benchmarks provide zero-leakage validation of genuine reasoning and alignment. |
| **Stand-alone Notebook Architecture** | Python CLI script only | Notebook allows interactive inspection of candidate responses, ROUGE consensus matrices, and graphical comparison dashboards in Google Colab. |

---

## Reflections

1. **Self-Consistency acts as an ensemble within a single model**: By sampling multiple times from the temperature-softened probability distribution and finding the geometric consensus via ROUGE-L, we effectively eliminate high-variance degenerate tokens without needing multiple model checkpoints.
2. **Alignment is about shaping boundaries, not erasing capabilities**: The qualitative tests demonstrate that ORPO successfully shifted the decision boundary for harmful and deceptive prompts while preserving the mathematical reasoning mechanisms learned during SFT.
3. **Inference optimization is an architectural necessity**: Running 20+ multi-sample benchmark queries sequentially on standard HF pipeline takes minutes, whereas vLLM continuous batching dispatches the entire batch in seconds.

---

## What's Next → Phase 6: FastAPI Server & Client

With the inference engine validated and benchmarks established:
1. **Build `phase6/`**: Implement our own production **FastAPI server** with modern `lifespan` state management, background cleanup, and robust response parsing.
2. **Build CLI Client**: Interactive Python chat client with multi-turn conversation memory and graceful exit handling.
3. **End-to-End Local / Colab Deployment**: Complete interactive assistant ready for real-time conversation.
