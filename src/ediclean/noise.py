from __future__ import annotations

import random
import re
import string
from typing import Dict, Iterable, List, Sequence

SENTIMENT_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "best", "love", "loved", "like", "liked",
    "bad", "terrible", "awful", "worst", "boring", "dull", "hate", "hated", "poor", "disappointing",
    "funny", "beautiful", "perfect", "horrible", "brilliant", "enjoyable", "unpleasant", "masterpiece",
    "waste", "mess", "charming", "moving", "annoying", "weak", "strong"
}

# Words that can usually be removed without intentionally flipping sentiment.
# Negation words are deliberately excluded because deleting "not" can change labels.
DEFAULT_FUNCTION_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "at", "by", "from", "as", "that",
    "this", "these", "those", "it", "its", "there", "here", "very", "really", "quite", "just", "so"
}

PROTECTED_WORDS = {
    "not", "no", "never", "n't", "cannot", "cant", "can't", "without", "hardly", "barely", "neither", "nor"
} | SENTIMENT_WORDS

KEYBOARD_NEIGHBORS = {
    "q": "was", "w": "qase", "e": "wsdr", "r": "edft", "t": "rfgy", "y": "tghu", "u": "yhji", "i": "ujko", "o": "iklp", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk"
}


def _char_swap(word: str) -> str:
    if len(word) < 4:
        return word
    i = random.randint(1, len(word) - 2)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _char_delete(word: str) -> str:
    if len(word) < 4:
        return word
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i + 1:]


def _char_insert(word: str) -> str:
    if len(word) < 3:
        return word
    i = random.randint(1, len(word) - 1)
    return word[:i] + word[i - 1] + word[i:]


def _keyboard_substitute(word: str) -> str:
    if len(word) < 4:
        return word
    chars = list(word)
    candidates = [i for i, ch in enumerate(chars) if ch.lower() in KEYBOARD_NEIGHBORS and 0 < i < len(chars) - 1]
    if not candidates:
        return word
    i = random.choice(candidates)
    old = chars[i]
    new = random.choice(KEYBOARD_NEIGHBORS[old.lower()])
    chars[i] = new.upper() if old.isupper() else new
    return "".join(chars)


def _repeat_char(word: str) -> str:
    if len(word) < 3:
        return word
    i = random.randint(1, len(word) - 2)
    return word[:i] + word[i] + word[i:]


def _punctuation_drop(text: str, prob: float) -> str:
    return "".join(ch for ch in text if not (ch in string.punctuation and random.random() < prob))


def _word_drop(tokens: List[str], eligible_indices: List[int], n_changes: int) -> List[str]:
    if not eligible_indices:
        return tokens
    drop = set(random.sample(eligible_indices, min(n_changes, len(eligible_indices))))
    return [tok for i, tok in enumerate(tokens) if i not in drop]


def _word_order(tokens: List[str], eligible_indices: List[int], n_changes: int) -> List[str]:
    tokens = tokens[:]
    candidates = [i for i in eligible_indices if i + 1 < len(tokens)]
    random.shuffle(candidates)
    for i in candidates[:n_changes]:
        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    return tokens


def _clean_alpha(tok: str) -> str:
    return re.sub(r"[^A-Za-z]", "", tok).lower()


def _perturb_word(word: str, noise_type: str) -> str:
    if noise_type == "char_swap":
        return _char_swap(word)
    if noise_type == "char_delete":
        return _char_delete(word)
    if noise_type == "char_insert":
        return _char_insert(word)
    if noise_type == "keyboard_substitute":
        return _keyboard_substitute(word)
    if noise_type == "repeat_char":
        return _repeat_char(word)
    return word


def _weighted_choice(weights: Dict[str, float]) -> str:
    items = list(weights.keys())
    vals = [float(weights[k]) for k in items]
    total = sum(vals)
    if total <= 0:
        return random.choice(items)
    vals = [v / total for v in vals]
    return random.choices(items, weights=vals, k=1)[0]


def _eligible_chunk_indices(chunks: Sequence[str], min_len: int, focus_sentiment: bool) -> List[int]:
    word_indices = [i for i, c in enumerate(chunks) if c.isalpha() and len(c) >= min_len]
    if focus_sentiment:
        sent = [i for i in word_indices if chunks[i].lower() in SENTIMENT_WORDS]
        if sent:
            return sent
    return word_indices


def _eligible_token_indices(tokens: Sequence[str], min_len: int, *, protect_semantics: bool = True) -> List[int]:
    eligible = []
    for i, t in enumerate(tokens):
        clean = _clean_alpha(t)
        if len(clean) < min_len:
            continue
        if protect_semantics and clean in PROTECTED_WORDS:
            continue
        eligible.append(i)
    return eligible


def _function_word_drop(text: str, cfg: Dict, n_changes: int) -> str:
    function_words = set(cfg.get("function_words", list(DEFAULT_FUNCTION_WORDS)))
    protected = set(cfg.get("protected_words", list(PROTECTED_WORDS)))
    tokens = text.split()
    eligible = []
    for i, t in enumerate(tokens):
        clean = _clean_alpha(t)
        if clean in function_words and clean not in protected:
            eligible.append(i)
    return " ".join(_word_drop(tokens, eligible, max(1, n_changes))) if eligible else text


def _apply_char_noise(text: str, op: str, cfg: Dict, n_changes: int) -> str:
    min_len = int(cfg.get("min_word_length", 4))
    focus_sentiment = bool(cfg.get("focus_sentiment_words", False))
    chunks = re.findall(r"\w+|[^\w\s]+|\s+", text, flags=re.UNICODE)
    word_indices = _eligible_chunk_indices(chunks, min_len, focus_sentiment)
    if not word_indices:
        return text
    chosen = random.sample(word_indices, min(n_changes, len(word_indices)))
    for idx in chosen:
        chunks[idx] = _perturb_word(chunks[idx], op)
    return "".join(chunks)


def _apply_word_drop(text: str, cfg: Dict, n_changes: int) -> str:
    min_len = int(cfg.get("min_word_length", 4))
    focus_sentiment = bool(cfg.get("focus_sentiment_words", False))
    tokens = text.split()
    eligible = _eligible_token_indices(tokens, min_len, protect_semantics=True)
    if focus_sentiment:
        s_eligible = [i for i in eligible if _clean_alpha(tokens[i]) in SENTIMENT_WORDS]
        if s_eligible:
            eligible = s_eligible
    return " ".join(_word_drop(tokens, eligible, n_changes)) if eligible else text


def _apply_word_order(text: str, cfg: Dict, n_changes: int) -> str:
    min_len = int(cfg.get("min_word_length", 4))
    tokens = text.split()
    eligible = _eligible_token_indices(tokens, min_len, protect_semantics=True)
    return " ".join(_word_order(tokens, eligible, n_changes)) if eligible else text


def add_noise_to_text(text: str, cfg: Dict) -> str:
    noise_type = cfg.get("type", "human_like")
    ratio = float(cfg.get("ratio", 0.2))
    min_len = int(cfg.get("min_word_length", 4))
    focus_sentiment = bool(cfg.get("focus_sentiment_words", False))
    punctuation_drop_prob = float(cfg.get("punctuation_drop_prob", 0.6))

    if not cfg.get("enabled", True):
        return text

    chunks = re.findall(r"\w+|[^\w\s]+|\s+", text, flags=re.UNICODE)
    word_indices = _eligible_chunk_indices(chunks, min_len, focus_sentiment)
    n_changes = max(1, int(round(max(len(word_indices), 1) * ratio)))

    if noise_type == "punctuation_drop":
        return _punctuation_drop(text, punctuation_drop_prob)
    if noise_type in {"char_swap", "char_delete", "char_insert", "keyboard_substitute", "repeat_char"}:
        return _apply_char_noise(text, noise_type, cfg, n_changes)
    if noise_type == "word_drop":
        return _apply_word_drop(text, cfg, n_changes)
    if noise_type == "word_order":
        return _apply_word_order(text, cfg, n_changes)
    if noise_type == "function_word_drop":
        return _function_word_drop(text, cfg, n_changes)
    if noise_type == "lowercase":
        return text.lower()

    if noise_type == "mixed":
        mixed_types = cfg.get("mixed_types", ["char_swap", "char_delete", "char_insert", "word_drop"])
        noisy = text
        # Apply at least one char-level operation, then optionally structural operations.
        char_types = [t for t in mixed_types if t in {"char_swap", "char_delete", "char_insert", "keyboard_substitute", "repeat_char"}]
        if char_types:
            noisy = _apply_char_noise(noisy, random.choice(char_types), cfg, n_changes)
        if "punctuation_drop" in mixed_types and random.random() < cfg.get("mixed_punctuation_prob", 0.5):
            noisy = _punctuation_drop(noisy, punctuation_drop_prob)
        if "word_drop" in mixed_types and random.random() < cfg.get("mixed_word_drop_prob", 0.5):
            noisy = _apply_word_drop(noisy, cfg, max(1, int(n_changes * 0.5)))
        if "word_order" in mixed_types and random.random() < cfg.get("mixed_word_order_prob", 0.35):
            noisy = _apply_word_order(noisy, cfg, 1)
        if "function_word_drop" in mixed_types and random.random() < cfg.get("mixed_function_word_drop_prob", 0.35):
            noisy = _function_word_drop(noisy, cfg, 1)
        return noisy

    if noise_type == "human_like":
        weights = cfg.get("human_like_weights", {
            "char_swap": 0.18,
            "char_delete": 0.17,
            "char_insert": 0.15,
            "keyboard_substitute": 0.10,
            "repeat_char": 0.10,
            "punctuation_drop": 0.10,
            "function_word_drop": 0.10,
            "word_order": 0.07,
            "lowercase": 0.03,
        })
        # Number of operations scales gently with ratio. For short SST-2 sentences, 1--2 ops is enough.
        n_ops = max(1, int(round(float(cfg.get("ops_multiplier", 6)) * ratio)))
        noisy = text
        for _ in range(n_ops):
            op = _weighted_choice(weights)
            if op in {"char_swap", "char_delete", "char_insert", "keyboard_substitute", "repeat_char"}:
                noisy = _apply_char_noise(noisy, op, cfg, 1)
            elif op == "punctuation_drop":
                noisy = _punctuation_drop(noisy, punctuation_drop_prob)
            elif op == "function_word_drop":
                noisy = _function_word_drop(noisy, cfg, 1)
            elif op == "word_drop":
                noisy = _apply_word_drop(noisy, cfg, 1)
            elif op == "word_order":
                noisy = _apply_word_order(noisy, cfg, 1)
            elif op == "lowercase":
                noisy = noisy.lower()
        return noisy

    # Default fallback: one of the character noise types.
    return _apply_char_noise(text, "char_swap", cfg, n_changes)


def add_noise_batch(texts: Iterable[str], cfg: Dict, seed: int) -> List[str]:
    random.seed(seed)
    return [add_noise_to_text(t, cfg) for t in texts]
