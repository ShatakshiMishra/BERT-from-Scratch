# 02 — BERT (from scratch)

A small, heavily-commented implementation of **BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding** (Devlin et al., 2018),
built in the same teaching style as `01-transformer`. It runs end-to-end on an
Apple-Silicon laptop (MPS) in a couple of minutes.

The goal is to *understand* BERT, not to reproduce BERT-Base. So the corpus is
Tiny Shakespeare, the tokenizer is word-level (not WordPiece), and the model is
tiny (4 layers, hidden 256, ~5M params vs. the paper's 12 layers / 768 / 110M).
Every mechanism from the paper is present and real.

## The one idea to hold onto

`01-transformer` is a **decoder**: a causal mask lets each token see only the
tokens to its *left* (good for generation). BERT is an **encoder**: no causal
mask, so each token sees the *whole* sequence — left and right. That
bidirectionality is the "B" in BERT, and it's why BERT can't be trained as a
plain next-token predictor (a token would see itself). The fix is the **Masked
LM** objective.

## Files

| File | What it is | Paper section |
|------|------------|---------------|
| `tokenizer.py` | Word-level tokenizer + the 5 special tokens `[PAD] [CLS] [SEP] [MASK] [UNK]` (a stand-in for WordPiece) | §3, Input Representations |
| `data.py` | Builds pretraining examples: NSP pairs + MLM masking (the 80/10/10 rule) | §3.1, Tasks #1 & #2 |
| `model.py` | Embeddings (token+segment+position), bidirectional multi-head attention, encoder stack, MLM head + NSP head | §3, Model Architecture |
| `pretrain.py` | Trains on MLM + NSP jointly; saves `bert_pretrained.pt` | §3.1 |
| `finetune.py` | Loads the pretrained backbone, adds one linear head on `[CLS]`, fine-tunes for a downstream classification task | §3.2, Fine-tuning |

## How to run

```bash
cd 02-bert
../01-transformer/.venv/bin/python pretrain.py   # pre-train (writes bert_pretrained.pt)
../01-transformer/.venv/bin/python finetune.py   # fine-tune + compare vs. from-scratch
```

(Each script also runs standalone as a sanity check, e.g. `python data.py`.)

## What each part demonstrates from the paper

- **Masked LM (§3.1).** `data.apply_mlm` picks 15% of tokens and replaces them
  80% → `[MASK]`, 10% → random token, 10% → unchanged. The 10/10 exist because
  `[MASK]` never appears at fine-tuning time, so training only on `[MASK]` would
  create a pre-train/fine-tune mismatch.
- **Next Sentence Prediction (§3.1).** `data.make_pair` builds 50% real / 50%
  random sentence pairs; the model classifies "IsNext?" from the `[CLS]` vector.
- **Input = token + segment + position (Figure 2).** `model.BertEmbeddings`.
- **Bidirectional attention.** `model.MultiHeadAttention` applies *only* a
  padding mask — never a causal one.
- **Pre-train → fine-tune recipe (Figure 1).** `pretrain.py` saves the backbone;
  `finetune.py` reloads it and fine-tunes *all* parameters with one new head.

## Results on this tiny setup

Pre-training (3000 steps): MLM loss ≈ 8.9 → ≈ 4.1, and masked-word predictions
become grammatically sensible (pronouns, verbs in the right slots).

Fine-tuning (question vs. statement, trailing punctuation removed so the model
must read the words):

| Init | val accuracy after 3 epochs |
|------|------|
| **from pre-trained backbone** | **~0.90** |
| from scratch | ~0.86 |

The pre-trained model is both better and faster to converge — a miniature
version of the paper's headline finding that unsupervised pre-training transfers.

## Honest simplifications (and what they teach)

- **Word-level, not WordPiece.** WordPiece adds subword splitting (`play` +
  `##ing`); it doesn't change any of the BERT machinery here.
- **Lines as "sentences" for NSP.** Shakespeare's short lines make NSP hard, so
  NSP accuracy here is weak. That's a genuine echo of later findings (RoBERTa)
  that NSP is of questionable value — MLM does the heavy lifting.
- **Tiny model / corpus.** Enough to see every mechanism work; not enough for
  the paper's benchmark numbers.
