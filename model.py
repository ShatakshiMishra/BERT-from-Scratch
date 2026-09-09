"""
model.py
========
The BERT model itself (paper §3, "Model Architecture").

BERT is a multi-layer *bidirectional* Transformer ENCODER. The single most
important contrast with what you built in 01-transformer (a GPT-style decoder):

    decoder  : causal mask -> each token sees only tokens to its LEFT
    encoder  : NO causal mask -> each token sees the ENTIRE sequence

That full, both-directions view is exactly what "B" (Bidirectional) in BERT
means, and it's why BERT needs the Masked-LM trick (§3.1): if every token could
see every other token *including itself*, predicting the next token would be
trivial, so instead we hide 15% of tokens and predict those.

We build, from scratch:
  1. BertEmbeddings   - token + segment + position embeddings, summed (Figure 2)
  2. MultiHeadAttention (bidirectional, with a padding mask)
  3. EncoderLayer     - attention + feed-forward, post-LayerNorm, GELU
  4. Bert             - the stack of encoder layers + a [CLS] "pooler"
  5. BertPreTraining  - Bert + the MLM head and the NSP head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    """Small config so the whole thing trains quickly on an MPS laptop.
    (Paper's BERT-Base is vocab=30k, hidden=768, layers=12, heads=12.)"""
    def __init__(self, vocab_size, max_len=64, hidden=256, layers=4, heads=4,
                 ff=1024, dropout=0.1):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.ff = ff
        self.dropout = dropout


# ---------------------------------------------------------------------------
# 1. Embeddings: the input vector for a token is the SUM of three embeddings.
#    (Paper Figure 2: token + segment + position.)
# ---------------------------------------------------------------------------
class BertEmbeddings(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.hidden)   # what the word is
        self.segment = nn.Embedding(2, cfg.hidden)              # sentence A vs B
        self.position = nn.Embedding(cfg.max_len, cfg.hidden)   # where it sits
        self.norm = nn.LayerNorm(cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, segment_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.token(input_ids) + self.segment(segment_ids) + self.position(pos)
        return self.drop(self.norm(x))


# ---------------------------------------------------------------------------
# 2. Multi-head self-attention. Bidirectional: the ONLY mask is for padding,
#    so real tokens can attend to every other real token, left and right.
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.hidden % cfg.heads == 0
        self.h = cfg.heads
        self.dk = cfg.hidden // cfg.heads
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden)   # project to Q, K, V at once
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)      # mix heads back together
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask):
        B, T, C = x.shape
        # Project and split into (B, heads, T, dk).
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        split = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        # Scaled dot-product scores: (B, heads, T, T).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)

        # Padding mask: forbid attending TO padding positions. attn_mask is
        # (B, T) True=real; reshape to (B, 1, 1, T) and set padded keys to -inf.
        pad = (~attn_mask).view(B, 1, 1, T)
        scores = scores.masked_fill(pad, float("-inf"))

        w = self.drop(F.softmax(scores, dim=-1))
        out = w @ v                                        # (B, heads, T, dk)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# 3. One encoder layer = self-attention + position-wise feed-forward, each
#    wrapped in a residual connection and LayerNorm (post-LN, as in the paper).
#    BERT uses the GELU activation in the feed-forward block.
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.hidden)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.ff),
            nn.GELU(),
            nn.Linear(cfg.ff, cfg.hidden),
        )
        self.norm2 = nn.LayerNorm(cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask):
        x = self.norm1(x + self.drop(self.attn(x, attn_mask)))   # residual + norm
        x = self.norm2(x + self.drop(self.ff(x)))                # residual + norm
        return x


# ---------------------------------------------------------------------------
# 4. The BERT backbone: embeddings -> stack of encoder layers -> two outputs.
#      sequence_output : (B, T, H) one vector per token  (used by MLM)
#      pooled_output   : (B, H)   the [CLS] vector, run through a tanh "pooler"
#                                 (used by NSP and by downstream classification)
# ---------------------------------------------------------------------------
class Bert(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embeddings = BertEmbeddings(cfg)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.layers))
        self.pooler = nn.Linear(cfg.hidden, cfg.hidden)   # for the [CLS] vector
        self.apply(self._init)                            # BERT-style init

    def _init(self, m):
        # Paper: weights ~ N(0, 0.02); biases 0; LayerNorm as default.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, segment_ids, attn_mask):
        x = self.embeddings(input_ids, segment_ids)
        for layer in self.layers:
            x = layer(x, attn_mask)
        sequence_output = x                               # (B, T, H)
        # The pooler takes the [CLS] token (position 0) and applies dense+tanh.
        pooled_output = torch.tanh(self.pooler(x[:, 0]))  # (B, H)
        return sequence_output, pooled_output


# ---------------------------------------------------------------------------
# 5. Pre-training model: BERT + the two heads used only during pre-training.
#      MLM head : project each token vector back to the vocabulary. Its output
#                 weights are TIED to the token embedding matrix (a common BERT
#                 trick that saves parameters and helps learning).
#      NSP head : a single linear layer on the pooled [CLS] vector -> 2 classes.
# ---------------------------------------------------------------------------
class BertPreTraining(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.bert = Bert(cfg)

        # MLM head: a small transform, then a decoder to vocab size.
        self.mlm_transform = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.LayerNorm(cfg.hidden)
        )
        self.mlm_decoder = nn.Linear(cfg.hidden, cfg.vocab_size, bias=True)
        # Weight tying: reuse the token embedding matrix as the output projection.
        self.mlm_decoder.weight = self.bert.embeddings.token.weight

        # NSP head: pooled [CLS] -> 2 logits (IsNext / NotNext).
        self.nsp = nn.Linear(cfg.hidden, 2)

    def forward(self, input_ids, segment_ids, attn_mask):
        seq, pooled = self.bert(input_ids, segment_ids, attn_mask)
        mlm_logits = self.mlm_decoder(self.mlm_transform(seq))  # (B, T, vocab)
        nsp_logits = self.nsp(pooled)                           # (B, 2)
        return mlm_logits, nsp_logits


if __name__ == "__main__":
    # Shape sanity check with random inputs.
    cfg = Config(vocab_size=6552)
    model = BertPreTraining(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.1f}M")
    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    seg = torch.zeros(B, T, dtype=torch.long)
    mask = torch.ones(B, T, dtype=torch.bool)
    mlm_logits, nsp_logits = model(ids, seg, mask)
    print("mlm_logits:", tuple(mlm_logits.shape), "| nsp_logits:", tuple(nsp_logits.shape))
