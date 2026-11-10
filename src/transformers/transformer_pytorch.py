"""Case 1: PyTorch Transformer reimplementation (from demo_transformer.ipynb)."""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.data_utils import (
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    UNK_IDX,
    Field,
    detokenize_sentence,
)
from transformers.metrics import bleu_score


class Embedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 200, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        seq_length = x.size(1)
        pe = self.pe[:, :seq_length].to(x.device)
        x = x + pe
        return self.dropout(x)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention as in demo_transformer.ipynb."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.attn: Optional[torch.Tensor] = None

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        scores, self.attn = attention(q, k, v, mask, self.dropout)
        self.viz_scores = self.attn
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(concat)


class Norm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True, unbiased=False) + self.eps)
            + self.bias
        )
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, d_ff: int = 2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, d_ff: int = 2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        e_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        trg_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        heads: int,
        dropout: float,
        d_ff: int = 2048,
        max_seq_length: int = 200,
    ):
        super().__init__()
        self.N = n_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_length=max_seq_length, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout, d_ff=d_ff), n_layers)
        self.norm = Norm(d_model)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        heads: int,
        dropout: float,
        d_ff: int = 2048,
        max_seq_length: int = 200,
    ):
        super().__init__()
        self.N = n_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_length=max_seq_length, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout, d_ff=d_ff), n_layers)
        self.norm = Norm(d_model)

    def forward(
        self,
        trg: torch.Tensor,
        e_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        trg_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        trg_vocab: int,
        d_model: int = 512,
        N: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
        max_seq_length: int = 200,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout, d_ff=d_ff, max_seq_length=max_seq_length)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, d_ff=d_ff, max_seq_length=max_seq_length)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        trg_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        return self.out(d_output)


def get_model(opt, src_vocab: int, trg_vocab: int) -> Transformer:
    """Create and Xavier-initialize a `Transformer` model from an options dict or namespace."""
    getter = opt.get if isinstance(opt, dict) else lambda k, default=None: getattr(opt, k, default)
    d_model = getter("d_model", 512)
    n_layers = getter("n_layers", 6)
    heads = getter("heads", 8)
    dropout = getter("dropout", 0.1)
    d_ff = getter("d_ff", 2048)
    max_seq_length = getter("max_seq_length", 200)

    assert d_model % heads == 0
    assert dropout < 1

    model = Transformer(
        src_vocab,
        trg_vocab,
        d_model=d_model,
        N=n_layers,
        heads=heads,
        dropout=dropout,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
    )
    load_weights = getter("load_weights", None)
    if load_weights is not None:
        state = torch.load(f"{load_weights}/model_weights", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    device = getter("device", None)
    if device is not None:
        model = model.to(device)
    return model


def nopeak_mask(size: int, device: torch.device | str) -> torch.Tensor:
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    mask = torch.from_numpy(np_mask) == 0
    return mask.to(device)


def create_masks(
    src: torch.Tensor,
    trg: Optional[torch.Tensor],
    src_pad: int = PAD_IDX,
    trg_pad: int = PAD_IDX,
    device: torch.device | str = "cuda",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    src_mask = (src != src_pad).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size, device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask


class ScheduledOptim:
    """Noam learning rate schedule wrapper as in demo_transformer.ipynb."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float = 0.2,
        d_model: int = 512,
        n_warmup_steps: int = 4000,
        lr_mul: Optional[float] = None,
    ):
        self._optimizer = optimizer
        self.init_lr = float(lr_mul if lr_mul is not None else init_lr)
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.current_lr = 0.0

    def step_and_update_lr(self) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> float:
        return (self.d_model ** -0.5) * min(
            self.n_steps ** (-0.5),
            self.n_steps * (self.n_warmup_steps ** (-1.5)),
        )

    def _update_learning_rate(self) -> None:
        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.current_lr = float(lr)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self) -> dict:
        return {
            "init_lr": self.init_lr,
            "d_model": self.d_model,
            "n_warmup_steps": self.n_warmup_steps,
            "n_steps": self.n_steps,
            "_optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.init_lr = state_dict["init_lr"]
        self.d_model = state_dict["d_model"]
        self.n_warmup_steps = state_dict["n_warmup_steps"]
        self.n_steps = state_dict["n_steps"]
        self._optimizer.load_state_dict(state_dict["_optimizer"])


class LabelSmoothingLoss(nn.Module):
    """Label smoothing KL loss ignoring `<pad>` tokens as in demo_transformer.ipynb."""

    def __init__(self, classes: int, padding_idx: int = PAD_IDX, smoothing: float = 0.1, dim: int = -1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / max(1, self.cls - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target == self.padding_idx, as_tuple=False)
            if mask.numel() > 0:
                true_dist.index_fill_(0, mask.squeeze(-1), 0.0)

        non_pad = (target != self.padding_idx).sum().clamp(min=1)
        return torch.sum(-true_dist * pred) / non_pad


def init_vars(
    src: torch.Tensor,
    model: Transformer,
    SRC: Field,
    TRG: Field,
    device: torch.device | str,
    k: int,
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    init_tok = TRG.vocab.stoi.get("<sos>", SOS_IDX)
    src_pad = SRC.vocab.stoi.get("<pad>", PAD_IDX)
    src_mask = (src != src_pad).unsqueeze(-2)

    e_output = model.encoder(src, src_mask)
    outputs = torch.tensor([[init_tok]], dtype=torch.long, device=device)
    trg_mask = nopeak_mask(1, device)

    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].topk(k)
    log_scores = torch.log(probs.clamp(min=1e-12)).cpu()

    outputs = torch.zeros(k, max_len, dtype=torch.long, device=device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = e_output.repeat(k, 1, 1)
    return outputs, e_outputs, log_scores


def k_best_outputs(
    outputs: torch.Tensor,
    out: torch.Tensor,
    log_scores: torch.Tensor,
    i: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs, ix = out[:, -1].topk(k)
    log_probs = torch.log(probs.clamp(min=1e-12)).cpu().view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    log_scores = k_probs.unsqueeze(0)
    return outputs, log_scores


@torch.no_grad()
def beam_search(
    src: torch.Tensor,
    model: Transformer,
    SRC: Field,
    TRG: Field,
    device: torch.device | str,
    k: int = 5,
    max_len: int = 80,
) -> str:
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, device, k, max_len)
    eos_tok = TRG.vocab.stoi.get("<eos>", EOS_IDX)
    src_pad = SRC.vocab.stoi.get("<pad>", PAD_IDX)
    src_mask = (src != src_pad).unsqueeze(-2)
    ind: Optional[int] = None

    for i in range(2, max_len):
        trg_mask = nopeak_mask(i, device)
        out = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, k)

        ones = (outputs == eos_tok).nonzero(as_tuple=False)
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long)
        for vec in ones:
            row_idx = int(vec[0].item())
            if sentence_lengths[row_idx] == 0:
                sentence_lengths[row_idx] = int(vec[1].item())

        num_finished_sentences = int((sentence_lengths > 0).sum().item())
        if num_finished_sentences == k:
            alpha = 0.7
            div = 1.0 / (sentence_lengths.type_as(log_scores) ** alpha)
            _, best_idx = torch.max(log_scores * div, 1)
            ind = int(best_idx.item())
            break

    chosen = outputs[0] if ind is None else outputs[ind]
    eos_positions = (chosen == eos_tok).nonzero(as_tuple=False)
    if len(eos_positions) > 0:
        length = int(eos_positions[0].item())
        tok_ids = chosen[1:length].tolist()
    else:
        tok_ids = chosen[1:].tolist()

    return " ".join(TRG.vocab.itos[tok] for tok in tok_ids if tok not in (PAD_IDX, SOS_IDX, EOS_IDX))


@torch.no_grad()
def translate_sentence(
    sentence: str,
    model: Transformer,
    SRC: Field,
    TRG: Field,
    device: torch.device | str,
    k: int = 5,
    max_len: int = 80,
) -> str:
    """Translate a single English sentence into Vietnamese using Beam Search."""
    model.eval()
    tokens = SRC.preprocess(sentence)
    if not tokens:
        return ""
    indexed = [SRC.vocab.stoi.get(tok, UNK_IDX) for tok in tokens]
    src_tensor = torch.tensor([indexed], dtype=torch.long, device=device)
    raw_translation = beam_search(src_tensor, model, SRC, TRG, device, k=k, max_len=max_len)
    return detokenize_sentence(raw_translation)


@torch.no_grad()
def greedy_decode_batch(
    src: torch.Tensor,
    model: Transformer,
    src_pad: int = PAD_IDX,
    sos_idx: int = SOS_IDX,
    eos_idx: int = EOS_IDX,
    max_len: int = 60,
) -> List[List[int]]:
    """Fast batched greedy decoding for corpus-level BLEU evaluation."""
    model.eval()
    device = src.device
    bs = src.size(0)
    src_mask = (src != src_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    ys = torch.full((bs, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(bs, dtype=torch.bool, device=device)

    for _ in range(1, max_len):
        trg_mask = nopeak_mask(ys.size(1), device)
        out = model.out(model.decoder(ys, e_outputs, src_mask, trg_mask))
        next_word = out[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_word], dim=1)
        finished |= next_word.squeeze(1) == eos_idx
        if bool(finished.all()):
            break

    results: List[List[int]] = []
    for row in ys[:, 1:].tolist():
        seq: List[int] = []
        for idx in row:
            if idx == eos_idx:
                break
            if idx not in (src_pad, sos_idx):
                seq.append(idx)
        results.append(seq)
    return results


@torch.no_grad()
def translate_corpus_batched(
    src_sentences: Sequence[str],
    model: Transformer,
    SRC: Field,
    TRG: Field,
    device: torch.device | str,
    batch_size: int = 64,
    max_len: int = 60,
) -> List[str]:
    """Translate a list of sentences in batches using greedy decoding."""
    model.eval()
    predictions: List[str] = []
    for start in range(0, len(src_sentences), batch_size):
        chunk = src_sentences[start : start + batch_size]
        encoded = [
            [SRC.vocab.stoi.get(tok, UNK_IDX) for tok in SRC.preprocess(s)[:max_len]] or [UNK_IDX]
            for s in chunk
        ]
        max_s = max(len(seq) for seq in encoded)
        src_tensor = torch.full((len(chunk), max_s), PAD_IDX, dtype=torch.long, device=device)
        for i, seq in enumerate(encoded):
            src_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        pred_ids = greedy_decode_batch(src_tensor, model, max_len=max_len)
        for ids in pred_ids:
            words = [TRG.vocab.itos[idx] for idx in ids if 0 <= idx < len(TRG.vocab)]
            predictions.append(detokenize_sentence(" ".join(words)))
    return predictions


def step(
    model: Transformer,
    optimizer: ScheduledOptim,
    batch,
    criterion: nn.Module,
    src_pad: int = PAD_IDX,
    trg_pad: int = PAD_IDX,
    device: torch.device | str = "cuda",
) -> float:
    """Single training step as in demo_transformer.ipynb."""
    model.train()
    src = batch.src.transpose(0, 1).to(device)
    trg = batch.trg.transpose(0, 1).to(device)
    trg_input = trg[:, :-1]
    src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, device)
    preds = model(src, trg_input, src_mask, trg_mask)

    ys = trg[:, 1:].contiguous().view(-1)
    optimizer.zero_grad()
    loss = criterion(preds.view(-1, preds.size(-1)), ys)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step_and_update_lr()
    return float(loss.item())


@torch.no_grad()
def validate(
    model: Transformer,
    valid_iter,
    criterion: nn.Module,
    src_pad: int = PAD_IDX,
    trg_pad: int = PAD_IDX,
    device: torch.device | str = "cuda",
) -> float:
    """Compute average validation loss across `valid_iter`."""
    model.eval()
    total_loss: List[float] = []
    for batch in valid_iter:
        src = batch.src.transpose(0, 1).to(device)
        trg = batch.trg.transpose(0, 1).to(device)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, device)
        preds = model(src, trg_input, src_mask, trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        loss = criterion(preds.view(-1, preds.size(-1)), ys)
        total_loss.append(float(loss.item()))
    return float(np.mean(total_loss)) if total_loss else 0.0


def bleu(
    valid_src_data: Sequence[str],
    valid_trg_data: Sequence[str],
    model: Transformer,
    SRC: Field,
    TRG: Field,
    device: torch.device | str,
    k: int = 5,
    max_strlen: int = 60,
    use_beam: bool = False,
) -> float:
    """Compute BLEU score on validation/test set (compatible with demo_transformer.ipynb)."""
    if use_beam:
        pred_sents = [
            translate_sentence(sent, model, SRC, TRG, device, k, max_strlen)
            for sent in valid_src_data
        ]
    else:
        pred_sents = translate_corpus_batched(
            valid_src_data, model, SRC, TRG, device, batch_size=64, max_len=max_strlen
        )

    pred_toks = [TRG.preprocess(sent) for sent in pred_sents]
    trg_toks = [[TRG.preprocess(sent)] for sent in valid_trg_data]
    return bleu_score(pred_toks, trg_toks)
