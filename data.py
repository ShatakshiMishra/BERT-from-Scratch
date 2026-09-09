"""
data.py
=======
This file turns raw text into the exact tensors BERT is pre-trained on. It
implements the two unsupervised tasks from the paper (§3.1):

  Task #1  Masked LM (MLM): mask 15% of tokens, predict them from BOTH sides.
  Task #2  Next Sentence Prediction (NSP): given sentence A and B, is B the
           real sentence that followed A, or a random one?

Every training example is the pair packed into one sequence, exactly as in the
paper's Figure 2:

    [CLS] sentence A tokens [SEP] sentence B tokens [SEP]

plus three parallel pieces of information the model needs:
  - segment ids  : 0 for the "[CLS] A [SEP]" part, 1 for the "B [SEP]" part
  - mlm labels   : the true id at masked positions, and -100 everywhere else
                   (-100 is PyTorch's default "ignore" index for the loss)
  - nsp label    : 1 if B really follows A ("IsNext"), else 0 ("NotNext")
"""

import random
import torch
from tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# 1. Load the corpus and split it into "sentences".
#    The paper uses document-level text so it can find the *actual* next
#    sentence. We treat each non-empty line as a sentence and remember order,
#    so "the next line" plays the role of "the next sentence".
# ---------------------------------------------------------------------------
with open("input.txt") as f:
    raw = f.read()

tokenizer = Tokenizer(raw)

# Encode every line once, up front. Keep only lines with >= 2 tokens.
sentences = [ids for line in raw.split("\n") if len(ids := tokenizer.encode(line)) >= 2]
n_sent = len(sentences)


def make_pair(max_len: int = 64):
    """Build ONE (A, B, is_next) sentence pair for the NSP task (paper §3.1)."""
    # Pick a sentence A that has a real following sentence available.
    i = random.randint(0, n_sent - 2)
    a = sentences[i]

    if random.random() < 0.5:
        # 50%: B is the genuine next sentence -> label IsNext (1)
        b, is_next = sentences[i + 1], 1
    else:
        # 50%: B is a random sentence -> label NotNext (0)
        j = random.randint(0, n_sent - 1)
        b, is_next = sentences[j], 0

    # Truncate so [CLS] A [SEP] B [SEP] fits in max_len (2 sentences + 3 specials).
    budget = max_len - 3
    a = a[: budget // 2]
    b = b[: budget - len(a)]
    return a, b, is_next


def apply_mlm(token_ids):
    """The Masked-LM masking scheme from the paper (§3.1).

    Choose 15% of positions. For each chosen position, replace it with:
        80% of the time -> [MASK]
        10% of the time -> a random token
        10% of the time -> leave it unchanged
    The 10% "random" and 10% "unchanged" cases exist because [MASK] never
    appears at fine-tuning time; this reduces the pre-train/fine-tune mismatch.

    Returns (possibly-corrupted input ids, labels) where labels hold the TRUE
    id at masked positions and -100 elsewhere (so the loss ignores them).
    """
    ids = token_ids.copy()
    labels = [-100] * len(ids)

    for pos in range(len(ids)):
        # Never mask the structural tokens [CLS] / [SEP].
        if ids[pos] in (tokenizer.cls_id, tokenizer.sep_id):
            continue
        if random.random() < 0.15:                # 15% of tokens are chosen
            labels[pos] = ids[pos]                # remember the answer first
            r = random.random()
            if r < 0.8:
                ids[pos] = tokenizer.mask_id       # 80% -> [MASK]
            elif r < 0.9:
                ids[pos] = random.randint(0, tokenizer.vocab_size - 1)  # 10% -> random
            # else: 10% -> keep the original token unchanged
    return ids, labels


def make_example(max_len: int = 64):
    """Assemble one fully-prepared training example (still plain Python lists)."""
    a, b, is_next = make_pair(max_len)

    # Pack into a single sequence with the special tokens (paper Figure 2).
    ids = [tokenizer.cls_id] + a + [tokenizer.sep_id] + b + [tokenizer.sep_id]
    # Segment ids: 0 covers "[CLS] A [SEP]", 1 covers "B [SEP]".
    segment = [0] * (len(a) + 2) + [1] * (len(b) + 1)

    # Corrupt with the MLM scheme and get the labels.
    ids, mlm_labels = apply_mlm(ids)
    return ids, segment, mlm_labels, is_next


def get_batch(batch_size: int, max_len: int = 64, device: str = "cpu"):
    """Build a padded batch of examples and return it as tensors.

    Shapes (B = batch_size, T = longest example in this batch):
      input_ids   (B, T) long   - token ids, with padding
      segment_ids (B, T) long   - 0 / 1 segment markers
      attn_mask   (B, T) bool   - True where there is a REAL token (not padding)
      mlm_labels  (B, T) long   - true ids at masked spots, -100 elsewhere
      nsp_labels  (B,)   long   - 1 = IsNext, 0 = NotNext
    """
    batch = [make_example(max_len) for _ in range(batch_size)]
    T = max(len(ids) for ids, _, _, _ in batch)         # pad to longest in batch

    def pad(seq, fill):
        return seq + [fill] * (T - len(seq))

    input_ids, segment_ids, attn_mask, mlm_labels, nsp_labels = [], [], [], [], []
    for ids, seg, lbl, nxt in batch:
        input_ids.append(pad(ids, tokenizer.pad_id))
        segment_ids.append(pad(seg, 0))
        attn_mask.append(pad([True] * len(ids), False))
        mlm_labels.append(pad(lbl, -100))               # padding is ignored too
        nsp_labels.append(nxt)

    t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    return (
        t(input_ids, torch.long),
        t(segment_ids, torch.long),
        t(attn_mask, torch.bool),
        t(mlm_labels, torch.long),
        t(nsp_labels, torch.long),
    )


if __name__ == "__main__":
    print("sentences:", n_sent, "| vocab:", tokenizer.vocab_size)
    input_ids, seg, mask, mlm, nsp = get_batch(batch_size=2, max_len=32)
    print("input_ids shape:", tuple(input_ids.shape))
    print("nsp labels:", nsp.tolist())
    # Show one example: what the model SEES vs what it must PREDICT.
    print("\nexample 0 (as the model sees it):")
    print(" ", tokenizer.decode(input_ids[0][mask[0]].tolist()))
    masked_positions = (mlm[0] != -100)
    print("masked answers it must recover:")
    print(" ", tokenizer.decode(mlm[0][masked_positions].tolist()))
