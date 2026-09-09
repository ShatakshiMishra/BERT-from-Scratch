"""
pretrain.py
===========
Pre-train BERT on our corpus with the paper's two objectives at once (§3.1):

    total loss = MLM loss (predict masked tokens)  +  NSP loss (IsNext?)

This is the whole point of the BERT paper: a SINGLE model, pre-trained on
unlabeled text with these two self-supervised tasks, learns representations you
can later fine-tune for many downstream tasks (see finetune.py).

Run:  ../01-transformer/.venv/bin/python pretrain.py
"""

import torch
import torch.nn.functional as F

from data import get_batch, tokenizer
from model import Config, BertPreTraining

# ---------------------------------------------------------------------------
# 1. Setup. MPS on Apple Silicon, else CPU.
# ---------------------------------------------------------------------------
torch.manual_seed(1337)
device = "mps" if torch.backends.mps.is_available() else "cpu"

MAX_LEN = 64
BATCH = 32
STEPS = 3000
EVAL_EVERY = 250

cfg = Config(vocab_size=tokenizer.vocab_size, max_len=MAX_LEN)
model = BertPreTraining(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
print(f"device: {device} | params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")


# ---------------------------------------------------------------------------
# 2. One training step: forward, combine the two losses, backprop.
# ---------------------------------------------------------------------------
def loss_on_batch():
    input_ids, seg, mask, mlm_labels, nsp_labels = get_batch(BATCH, MAX_LEN, device)
    mlm_logits, nsp_logits = model(input_ids, seg, mask)

    # MLM loss: cross-entropy over the vocab, but ONLY at masked positions.
    # cross_entropy ignores label == -100 automatically, so padded/unmasked
    # positions contribute nothing.
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, cfg.vocab_size), mlm_labels.view(-1), ignore_index=-100
    )
    # NSP loss: a plain 2-way classification on the [CLS] vector.
    nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
    return mlm_loss, nsp_loss


# ---------------------------------------------------------------------------
# 3. Train.
# ---------------------------------------------------------------------------
model.train()
for step in range(1, STEPS + 1):
    mlm_loss, nsp_loss = loss_on_batch()
    loss = mlm_loss + nsp_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilize training
    opt.step()

    if step % EVAL_EVERY == 0 or step == 1:
        print(f"step {step:4d} | mlm {mlm_loss.item():.3f} | nsp {nsp_loss.item():.3f}")


# ---------------------------------------------------------------------------
# 4. Demo: hand BERT a masked sentence and see what it fills in.
# ---------------------------------------------------------------------------
@torch.no_grad()
def fill_masks(text_a, text_b, mask_words):
    """Encode A/B, replace the given words with [MASK], show top predictions."""
    model.eval()
    a, b = tokenizer.encode(text_a), tokenizer.encode(text_b)
    ids = [tokenizer.cls_id] + a + [tokenizer.sep_id] + b + [tokenizer.sep_id]
    seg = [0] * (len(a) + 2) + [1] * (len(b) + 1)

    masked_positions = []
    for pos, tid in enumerate(ids):
        if tokenizer.itos[tid] in mask_words:
            ids[pos] = tokenizer.mask_id
            masked_positions.append(pos)

    t = lambda x: torch.tensor([x], device=device)
    mlm_logits, nsp_logits = model(t(ids), t(seg), t([True] * len(ids)))

    print("\ninput:", tokenizer.decode(ids))
    for pos in masked_positions:
        top5 = mlm_logits[0, pos].topk(5).indices.tolist()
        print(f"  [MASK] @ {pos} -> {[tokenizer.itos[i] for i in top5]}")
    is_next = torch.softmax(nsp_logits[0], -1)[1].item()
    print(f"  P(B follows A) = {is_next:.2f}")


# A real adjacent pair from the corpus vs. the same A with a masked word.
fill_masks("we know't, we know't.", "let us kill him", mask_words={"kill", "us"})
fill_masks("first citizen :", "before we proceed any further", mask_words={"proceed"})

# ---------------------------------------------------------------------------
# 5. Save the pre-trained backbone so finetune.py can load it. This mirrors the
#    paper's core recipe: pre-train ONCE on unlabeled text, then reuse those
#    parameters to initialize many downstream models (paper §3, Figure 1).
# ---------------------------------------------------------------------------
torch.save(model.bert.state_dict(), "bert_pretrained.pt")
print("\nsaved pre-trained backbone -> bert_pretrained.pt")
