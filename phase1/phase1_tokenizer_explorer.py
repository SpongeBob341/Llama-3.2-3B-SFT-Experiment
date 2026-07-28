"""
=============================================================================
PHASE 1 — TOKENIZER EXPLORER
=============================================================================

Goal: Understand HOW Llama-3.2-3B sees text before you touch any training code.
      Everything in LLM training — data formatting, sequence length, batching —
      depends on understanding tokenization first.

What you'll learn:
  1. What a token is (not a word, not a character)
  2. How the tokenizer converts text → numbers (and back)
  3. What special tokens are and why they exist
  4. What Llama's chat template looks like (the format you'll train on)
  5. Why sequence length matters for training cost

Setup (run once in your terminal):
  pip install transformers torch huggingface_hub

Then log in to HuggingFace (needed for Llama access):
  huggingface-cli login
  (paste your HF token from https://huggingface.co/settings/tokens)

Run:
  python phase1_tokenizer_explorer.py
=============================================================================
"""

from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Load the tokenizer
# ─────────────────────────────────────────────────────────────────────────────
# We only load the TOKENIZER here — not the full model (which is 6GB+).
# The tokenizer alone is tiny (~a few MB) and runs on CPU instantly.
# This is how you do data preparation: load the tokenizer, format your dataset,
# never touch the actual model weights until training time on Colab.

print("=" * 70)
print("Loading Llama-3.2-3B tokenizer from HuggingFace...")
print("(This downloads ~3MB of tokenizer files, cached after first run)")
print("=" * 70)

MODEL_ID = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"\n✅ Tokenizer loaded!")
print(f"   Vocab size     : {tokenizer.vocab_size:,} tokens")
print(f"   Model max len  : {tokenizer.model_max_length:,} tokens")
print(f"   BOS token      : '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
print(f"   EOS token      : '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — What is a token?
# ─────────────────────────────────────────────────────────────────────────────
# Key insight: tokens are NOT words. The tokenizer uses Byte-Pair Encoding (BPE).
# Common words → 1 token. Rare words → split into multiple tokens.
# This matters because you pay for EVERY token during training (compute + memory).

print("\n" + "=" * 70)
print("SECTION 1: What is a token?")
print("=" * 70)

examples = [
    "Hello, how are you?",                        # simple, common words
    "supercalifragilisticexpialidocious",          # rare long word
    "2 + 2 = 4",                                  # math
    "def fibonacci(n): return n if n <= 1 else",  # code
    "नमस्ते दुनिया",                                # Hindi (non-Latin script)
    "🤖🔥💡",                                       # emoji
]

for text in examples:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"\n  Input   : {repr(text)}")
    print(f"  Tokens  : {tokens}")
    print(f"  IDs     : {token_ids}")
    print(f"  Count   : {len(token_ids)} tokens")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Encoding and Decoding
# ─────────────────────────────────────────────────────────────────────────────
# This is the actual operation that happens on every training sample:
#   text (string) → token IDs (list of ints) → fed into the model
#   model output (list of ints) → text (string) → shown to user

print("\n" + "=" * 70)
print("SECTION 2: Encoding → IDs → Decoding (the full round trip)")
print("=" * 70)

text = "The capital of France is Paris."
print(f"\n  Original text : '{text}'")

# Encode (text → IDs)
token_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"  Token IDs     : {token_ids}")

# Decode each token individually so you see the mapping
print(f"  Token by token:")
for i, (tid, tok) in enumerate(zip(token_ids, tokenizer.convert_ids_to_tokens(token_ids))):
    decoded = tokenizer.decode([tid])
    print(f"    [{i:2d}] ID {tid:6d} → token '{tok}' → decoded '{repr(decoded)}'")

# Full decode back to text
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
print(f"\n  Decoded back  : '{decoded_text}'")
print(f"  Round-trip OK : {text == decoded_text}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Special Tokens
# ─────────────────────────────────────────────────────────────────────────────
# Special tokens are NOT regular text. They're signals to the model.
# <|begin_of_text|> = start of a sequence
# <|end_of_text|>   = model should STOP generating here
# During SFT training, you place <|end_of_text|> at the end of every
# assistant response so the model learns WHEN to stop talking.

print("\n" + "=" * 70)
print("SECTION 3: Special tokens — the signals the model learns to obey")
print("=" * 70)

all_special = tokenizer.all_special_tokens
print(f"\n  All special tokens ({len(all_special)} total):")
for tok in all_special:
    tid = tokenizer.convert_tokens_to_ids(tok)
    print(f"    '{tok}'  →  ID {tid}")

print(f"""
  Why this matters for training:
    - Every training sample ends with '{tokenizer.eos_token}'
    - The model learns: "when I see this token was the target, stop generating"
    - Without it, the model would ramble on indefinitely at inference time
    - You'll add this to every formatted training sample in Phase 2
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Sequence Length & Training Cost
# ─────────────────────────────────────────────────────────────────────────────
# Attention is O(n²) in sequence length. Doubling the sequence length
# quadruples the memory used. This is why max_seq_length is a critical
# hyperparameter — it directly controls your GPU memory budget.

print("\n" + "=" * 70)
print("SECTION 4: Sequence length — why it dominates your training budget")
print("=" * 70)

seq_examples = [
    ("A short answer", "short"),
    ("What is the capital of France? The capital of France is Paris.", "Q+A"),
    (
        "Instruction:\nSolve the following math problem step by step.\n\n"
        "Input:\nIf a train travels 60 miles per hour for 2.5 hours, how far does it travel?\n\n"
        "Response:\nDistance = speed × time = 60 × 2.5 = 150 miles.",
        "full training sample"
    ),
]

print()
for text, label in seq_examples:
    ids = tokenizer.encode(text, add_special_tokens=True)
    print(f"  [{label}]")
    print(f"    Tokens: {len(ids)}")
    print(f"    Text  : {repr(text[:80])}{'...' if len(text) > 80 else ''}")
    print()

print(f"""  Rule of thumb for Colab training:
    max_seq_length = 512   → safe on T4 (15GB) with batch_size=4
    max_seq_length = 1024  → needs A100 or gradient checkpointing
    max_seq_length = 2048  → original project used this; needs A100 (80GB)
    
    We'll use 512 for our small validation run, then 1024 or 2048 for full run.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — The Chat Template (what your training data will look like)
# ─────────────────────────────────────────────────────────────────────────────
# Llama-3.2-3B uses a specific conversation format baked into the tokenizer.
# You MUST format your training samples to match this — otherwise you're
# training the model on a format it wasn't pre-trained to understand.

print("\n" + "=" * 70)
print("SECTION 5: Llama's chat template — the format your training data must use")
print("=" * 70)

# Check if Llama-3.2-3B has a built-in chat template
if tokenizer.chat_template is not None:
    print("\n  ✅ Llama-3.2-3B has a built-in chat template")

    # Format a single training example
    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = 4."},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # False when training (we include the answer)
    )

    print(f"\n  Input messages:")
    for m in messages:
        print(f"    [{m['role']}]: {m['content']}")

    print(f"\n  Formatted output (what actually goes into the model):")
    print("  " + "-" * 60)
    print(formatted)
    print("  " + "-" * 60)

    token_count = len(tokenizer.encode(formatted))
    print(f"\n  Token count for this sample: {token_count}")

    # Now show inference format (no answer, open-ended)
    inference_formatted = tokenizer.apply_chat_template(
        messages[:2],                # only system + user, no assistant
        tokenize=False,
        add_generation_prompt=True,  # True at inference (model continues from here)
    )
    print(f"\n  Inference format (add_generation_prompt=True, no answer):")
    print("  " + "-" * 60)
    print(inference_formatted)
    print("  " + "-" * 60)
    print("  ↑ At inference, the model sees everything up to here and generates the rest")

else:
    # Base model may not have a chat template — use the custom format from the original project
    print("\n  ⚠️  No built-in chat template found (expected for base model).")
    print("  We'll use the custom Alpaca-style format from the original project:\n")

    def format_training_sample(instruction, response, input_text=None):
        """
        The prompt format from the original Llama-3.2-3B SFT project.
        This is what every training sample will look like.
        """
        eos = tokenizer.eos_token  # <|end_of_text|>
        if input_text:
            return (
                f"Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{response}{eos}"
            )
        else:
            return (
                f"Instruction:\n{instruction}\n\n"
                f"### Response:\n{response}{eos}"
            )

    sample = format_training_sample(
        instruction="What is 2 + 2?",
        response="2 + 2 = 4.",
    )
    print(f"  Formatted sample:\n")
    print("  " + "-" * 60)
    print(sample)
    print("  " + "-" * 60)
    token_count = len(tokenizer.encode(sample))
    print(f"\n  Token count for this sample: {token_count}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — What you just learned (summary)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY — What you just learned")
print("=" * 70)
print("""
  ✅ Tokens are NOT words. They're subword units from BPE.
     Common words = 1 token. Rare words/numbers/code = multiple tokens.

  ✅ The tokenizer converts text → integers (encode) and back (decode).
     This is the first thing that happens to every training sample.

  ✅ Special tokens like <|end_of_text|> are signals, not words.
     You'll add EOS to the end of every assistant response during training.

  ✅ Sequence length is your biggest training cost lever.
     Longer sequences = more memory = smaller batch = slower training.
     Start with max_seq_length=512, scale up after validation.

  ✅ The chat template defines the exact format your training data must follow.
     Mismatch here = model trains on garbage = bad results.

─────────────────────────────────────────────────────────────────────────────
  NEXT: Phase 2 — Data Pipeline
  You'll load the 10 datasets, write a formatting function for each,
  and convert them all to the format you just saw above.
─────────────────────────────────────────────────────────────────────────────
""")
