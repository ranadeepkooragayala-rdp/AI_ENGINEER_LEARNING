"""
Day 9: Tokenization Internals, BPE, and Context Window Management
Topics Covered:
1. Byte-Pair Encoding (BPE) from scratch (Pair counting, iterative merging)
2. Production Tokenization using tiktoken (cl100k_base vs o200k_base, byte inspection)
3. ChatML Token Overhead Accounting
4. Dynamic Context Window Pruning & Token Budgeting
"""

from collections import Counter
import tiktoken


# ==========================================
# 1. BPE IMPLEMENTATION FROM SCRATCH
# ==========================================

def get_pair_frequencies(
    vocab: dict[tuple[str, ...], int],
) -> Counter[tuple[str, str]]:
    """Counts adjacent symbol pairs weighted by corpus word frequencies."""
    pairs: Counter[tuple[str, str]] = Counter()
    for tokens, freq in vocab.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += freq
    return pairs


def merge_vocab_pair(
    best_pair: tuple[str, str], vocab: dict[tuple[str, ...], int]
) -> dict[tuple[str, ...], int]:
    """Replaces all adjacent occurrences of best_pair with their merged form."""
    replacement = "".join(best_pair)
    new_vocab: dict[tuple[str, ...], int] = {}

    for tokens, freq in vocab.items():
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == best_pair[0]
                and tokens[i + 1] == best_pair[1]
            ):
                new_tokens.append(replacement)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_vocab[tuple(new_tokens)] = freq

    return new_vocab


def train_mini_bpe(
    raw_words: list[str], num_merges: int = 5
) -> tuple[dict[tuple[str, ...], int], list[tuple[str, str]]]:
    """Builds character-level vocabulary and learns iterative merge rules."""
    word_counts = Counter(raw_words)
    vocab = {
        tuple(list(word) + ["</w>"]): count for word, count in word_counts.items()
    }
    merges: list[tuple[str, str]] = []

    print("Initial Vocabulary State:")
    for tokens, freq in vocab.items():
        print(f"  {' '.join(tokens)} : {freq}")

    for step in range(num_merges):
        pairs = get_pair_frequencies(vocab)
        if not pairs:
            break

        best_pair, best_freq = pairs.most_common(1)[0]
        merges.append(best_pair)
        vocab = merge_vocab_pair(best_pair, vocab)

        print(f"\nMerge #{step + 1}: {best_pair} (occurred {best_freq} times)")
        print(f"  Merged token created: '{''.join(best_pair)}'")

    return vocab, merges


# ==========================================
# 2. PRODUCTION TOKENIZATION WITH TIKTOKEN
# ==========================================

def inspect_token_boundaries(text: str, encoding_name: str = "cl100k_base") -> None:
    """Decodes token IDs into individual raw byte representations."""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    print(f"\nText: '{text}'")
    print(f"Encoding: {encoding_name} | Total Tokens: {len(tokens)}")
    print(f"{'Token ID':<10} | {'Raw Bytes':<25} | {'Decoded Chunk'}")
    print("-" * 55)
    for token_id in tokens:
        token_bytes = enc.decode_single_token_bytes(token_id)
        decoded_repr = repr(token_bytes.decode("utf-8", errors="replace"))
        print(f"{token_id:<10} | {str(token_bytes):<25} | {decoded_repr}")


# ==========================================
# 3. CHATML TOKEN ACCOUNTING & PRUNING
# ==========================================

def calculate_chat_token_count(
    messages: list[dict[str, str]], model: str = "gpt-4o"
) -> int:
    """Calculates exact token usage for chat completions according to ChatML spec."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    tokens_per_name = 1
    total_tokens = 0

    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
            if key == "name":
                total_tokens += tokens_per_name

    total_tokens += 3  # Assistant reply primer
    return total_tokens


def prune_conversation_history(
    messages: list[dict[str, str]],
    max_token_budget: int = 100,
    model: str = "gpt-4o",
) -> list[dict[str, str]]:
    """Prunes chat history to fit within a strict token budget.
    
    Guarantees:
    - Preserves system prompt (if present).
    - Preserves newest user message.
    - Preserves recent turns via reverse-walk until token budget is exhausted.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    def count_msg_tokens(msg: dict[str, str]) -> int:
        return 3 + len(enc.encode(msg["role"])) + len(enc.encode(msg["content"]))

    if not messages:
        return []

    system_msg = messages[0] if messages[0].get("role") == "system" else None
    working_history = messages[1:] if system_msg else messages[:]

    if not working_history:
        return [system_msg] if system_msg else []

    latest_msg = working_history[-1]
    middle_history = working_history[:-1]

    # Non-negotiable base footprint
    base_tokens = 3  # Assistant reply primer
    if system_msg:
        base_tokens += count_msg_tokens(system_msg)
    base_tokens += count_msg_tokens(latest_msg)

    if base_tokens > max_token_budget:
        raise ValueError(
            f"Base system prompt + latest message ({base_tokens} tokens) "
            f"exceeds total budget ({max_token_budget} tokens)."
        )

    remaining_budget = max_token_budget - base_tokens

    # Reverse-walk through intermediate conversation turns
    retained_middle = []
    for msg in reversed(middle_history):
        cost = count_msg_tokens(msg)
        if cost <= remaining_budget:
            retained_middle.append(msg)
            remaining_budget -= cost
        else:
            break

    retained_middle.reverse()

    final_messages = []
    if system_msg:
        final_messages.append(system_msg)
    final_messages.extend(retained_middle)
    final_messages.append(latest_msg)

    return final_messages


# ==========================================
# 4. EXECUTION DEMO
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("1. RUNNING MINI-BPE TRAINING")
    print("=" * 60)
    sample_corpus = (
        ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
    )
    final_vocab, learned_merges = train_mini_bpe(sample_corpus, num_merges=4)

    print("\n=" * 60)
    print("2. TIKTOKEN INSPECTION & WHITESPACE SENSITIVITY")
    print("=" * 60)
    inspect_token_boundaries("Hello world! Coding in Python.")

    print("\n=" * 60)
    print("3. CONVERSATION TOKEN BUDGETING & PRUNING")
    print("=" * 60)
    conversation = [
        {"role": "system", "content": "You are a backend AI assistant with strict JSON output rules."},
        {"role": "user", "content": "Turn 1: What is a vector norm?"},
        {"role": "assistant", "content": "Turn 1: A vector norm measures vector length or magnitude."},
        {"role": "user", "content": "Turn 2: What is dot product?"},
        {"role": "assistant", "content": "Turn 2: Dot product multiplies matching dimensions and sums them."},
        {"role": "user", "content": "Turn 3: How do they relate to cosine similarity?"},
    ]

    total_tokens = calculate_chat_token_count(conversation, model="gpt-4o")
    print(f"Total unpruned conversation tokens: {total_tokens}")

    # Prune with a budget that forces older turns out
    budget = 65
    pruned_conversation = prune_conversation_history(conversation, max_token_budget=budget, model="gpt-4o")
    pruned_tokens = calculate_chat_token_count(pruned_conversation, model="gpt-4o")

    print(f"Target Budget: {budget} tokens | Pruned Payload Tokens: {pruned_tokens}")
    print(f"Original messages: {len(conversation)} -> Pruned messages: {len(pruned_conversation)}")
    print("\nPreserved Messages in Context:")
    for msg in pruned_conversation:
        print(f"  [{msg['role'].upper()}]: {msg['content']}")
