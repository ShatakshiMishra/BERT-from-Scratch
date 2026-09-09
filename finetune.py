"""
finetune.py
===========
The other half of the BERT paper (§3.2, "Fine-tuning BERT"): take the SAME
pre-trained model, add ONE small output layer, and fine-tune ALL parameters on
a labeled downstream task. The paper does this for GLUE, SQuAD, SWAG, etc.; the
recipe is always the same and always cheap.

Downstream task here: **question vs. statement** — does a line end with "?"
(label 1) or "." (label 0)? The labels come for free from the punctuation, so
it's a real, self-contained classification task. We use the [CLS] vector as the
"aggregate sequence representation" and put a linear classifier on top, exactly
as the paper does for sentence classification.

To show WHY pre-training matters, we fine-tune twice and compare:
    (a) starting from the pre-trained backbone (bert_pretrained.pt)
    (b) starting from random weights (no pre-training)

Run:  first `python pretrain.py` (creates bert_pretrained.pt), then this file.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import raw
from tokenizer import Tokenizer
from model import Config, Bert

torch.manual_seed(0)
random.seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_LEN = 64

# Rebuild the SAME tokenizer the backbone was pre-trained with (same corpus,
# same settings -> identical vocab and ids).
tokenizer = Tokenizer(raw)
cfg = Config(vocab_size=tokenizer.vocab_size, max_len=MAX_LEN)

# ---------------------------------------------------------------------------
# 1. Build the labeled dataset: keep lines ending in "?" or "." only.
#    IMPORTANT: we STRIP the final "?"/"." from the text. Otherwise the label
#    would literally be sitting in the input as a token and the model could
#    "cheat" by spotting it — the task would be trivial and pre-training would
#    show no benefit. Removing it forces the model to read the actual words
#    (e.g. questions tend to start with "what", "wilt", "why", "art").
# ---------------------------------------------------------------------------
examples = []
for line in raw.split("\n"):
    line = line.strip()
    if line.endswith("?"):
        examples.append((line[:-1].strip(), 1))
    elif line.endswith("."):
        examples.append((line[:-1].strip(), 0))
examples = [(t, y) for t, y in examples if t]   # drop any now-empty lines
random.shuffle(examples)
split = int(0.9 * len(examples))
train_ex, val_ex = examples[:split], examples[split:]
print(f"labeled examples: {len(examples)} (train {len(train_ex)} / val {len(val_ex)})")


def batchify(pairs, bs):
    """Yield padded (input_ids, segment_ids, attn_mask, labels) batches.
    Single-sentence input, so every segment id is 0 (paper: a text-∅ pair)."""
    for i in range(0, len(pairs), bs):
        chunk = pairs[i : i + bs]
        enc = [([tokenizer.cls_id] + tokenizer.encode(t)[: MAX_LEN - 2] + [tokenizer.sep_id], y)
               for t, y in chunk]
        T = max(len(ids) for ids, _ in enc)
        input_ids, mask, labels = [], [], []
        for ids, y in enc:
            input_ids.append(ids + [tokenizer.pad_id] * (T - len(ids)))
            mask.append([True] * len(ids) + [False] * (T - len(ids)))
            labels.append(y)
        seg = torch.zeros(len(enc), T, dtype=torch.long, device=device)
        t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
        yield t(input_ids, torch.long), seg, t(mask, torch.bool), t(labels, torch.long)


# ---------------------------------------------------------------------------
# 2. The fine-tuning model: BERT backbone + a linear head on the [CLS] vector.
#    (Paper §4.1: "the only new parameters introduced during fine-tuning are
#    classification layer weights".)
# ---------------------------------------------------------------------------
class BertClassifier(nn.Module):
    def __init__(self, cfg, n_classes=2):
        super().__init__()
        self.bert = Bert(cfg)
        self.classifier = nn.Linear(cfg.hidden, n_classes)

    def forward(self, input_ids, seg, mask):
        _, pooled = self.bert(input_ids, seg, mask)   # pooled = [CLS] vector
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# 3. Train / eval helpers.
# ---------------------------------------------------------------------------
@torch.no_grad()
def accuracy(model):
    model.eval()
    correct = total = 0
    for input_ids, seg, mask, y in batchify(val_ex, 64):
        pred = model(input_ids, seg, mask).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def run(pretrained: bool, epochs=3):
    model = BertClassifier(cfg).to(device)
    if pretrained:
        # Load the pre-trained backbone. strict=False because the pooler/heads
        # differ; the encoder + embeddings load exactly.
        sd = torch.load("bert_pretrained.pt", map_location=device)
        missing = model.bert.load_state_dict(sd, strict=False)
        print("  loaded pre-trained backbone", f"(missing: {len(missing.missing_keys)} keys)")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(train_ex)
        for input_ids, seg, mask, y in batchify(train_ex, 32):
            loss = F.cross_entropy(model(input_ids, seg, mask), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"  epoch {ep}: val accuracy = {accuracy(model):.3f}")


# ---------------------------------------------------------------------------
# 4. Compare: pre-trained initialization vs. from scratch.
# ---------------------------------------------------------------------------
print("\n=== fine-tuning FROM PRE-TRAINED backbone ===")
run(pretrained=True)

print("\n=== fine-tuning FROM SCRATCH (no pre-training) ===")
run(pretrained=False)
