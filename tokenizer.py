"""
tokenizer.py
============
BERT (paper §3, "Input/Output Representations") uses *WordPiece* with a 30,000
token vocabulary. WordPiece is a subword algorithm: it splits rare words into
pieces ("playing" -> "play" + "##ing") so the vocabulary can stay small while
still covering any word.

For this teaching implementation we use a much simpler *word-level* tokenizer.
The mechanics that matter for understanding BERT are identical — the only thing
we drop is subword splitting. We keep the FIVE special tokens BERT relies on:

    [PAD]   - filler so every sequence in a batch has the same length
    [CLS]   - prepended to every example; its final hidden state is the
              "aggregate" vector used for classification / NSP (paper §3)
    [SEP]   - separates sentence A from sentence B (and ends the sequence)
    [MASK]  - the token we swap in for the Masked-LM objective (paper §3.1)
    [UNK]   - any word not in our vocabulary
"""

import re

# The special tokens MUST get the first (lowest) ids so their meaning is fixed.
PAD, CLS, SEP, MASK, UNK = "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"
SPECIALS = [PAD, CLS, SEP, MASK, UNK]


def words(line: str):
    """Lowercase, then pull out words and standalone punctuation.

    "You'll die!" -> ["you", "'", "ll", "die", "!"]
    Simple and deterministic — good enough to teach the BERT machinery.
    """
    return re.findall(r"[a-z]+|[^a-z\s]", line.lower())


class Tokenizer:
    def __init__(self, text: str, max_vocab: int = 8000, min_freq: int = 2):
        # 1. Count how often each word appears in the whole corpus.
        freq = {}
        for w in words(text):
            freq[w] = freq.get(w, 0) + 1

        # 2. Keep the most common words (above min_freq), capped at max_vocab.
        #    Sort by frequency (desc) so the vocab is stable and meaningful.
        keep = sorted(
            (w for w, c in freq.items() if c >= min_freq),
            key=lambda w: (-freq[w], w),
        )[: max_vocab - len(SPECIALS)]

        # 3. Build the two lookup tables. Specials come first (ids 0..4).
        self.itos = SPECIALS + keep
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        # 4. Cache the ids of the special tokens for convenience.
        self.pad_id = self.stoi[PAD]
        self.cls_id = self.stoi[CLS]
        self.sep_id = self.stoi[SEP]
        self.mask_id = self.stoi[MASK]
        self.unk_id = self.stoi[UNK]
        self.vocab_size = len(self.itos)

    def encode(self, line: str):
        """A sentence of text -> a list of token ids (unknown words -> [UNK])."""
        return [self.stoi.get(w, self.unk_id) for w in words(line)]

    def decode(self, ids):
        """A list of token ids -> a readable string."""
        return " ".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    # Quick sanity check, in the same spirit as 01-transformer/data.py
    with open("input.txt") as f:
        text = f.read()
    tok = Tokenizer(text)
    print("vocab_size:", tok.vocab_size)
    print("special ids:", {s: tok.stoi[s] for s in SPECIALS})
    sample = "Let us kill him, and we'll have corn."
    ids = tok.encode(sample)
    print("encode:", ids)
    print("decode:", tok.decode(ids))
