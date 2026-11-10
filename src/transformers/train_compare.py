"""End-to-end training, evaluation, and visual comparison of:
- Case 1: PyTorch Transformer (from demo_transformer.ipynb / Vaswani et al., 2017)
- Case 2: TensorFlow/Keras Seq2Seq Bi-LSTM + Scaled Luong Attention (from tensorflow/nmt)
on the IWSLT 2015 English-Vietnamese Parallel Corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("KERAS_BACKEND", "torch")
warnings.filterwarnings("ignore", message=".*RNN module weights are not part of single contiguous chunk.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from tensorflow import keras

from transformers.data_utils import (
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    UNK_IDX,
    Field,
    ParallelTranslationDataset,
    Vocab,
    create_dataloader,
    detokenize_sentence,
    ensure_iwslt15_data,
    read_parallel_data,
)
from transformers.metrics import bleu_score, compute_translation_metrics
from transformers.nmt_tensorflow import (
    Seq2SeqNMT,
    evaluate_nmt_loss,
    masked_nmt_loss,
    translate_corpus_nmt_batched,
    translate_sentence_nmt,
)
from transformers.transformer_pytorch import (
    LabelSmoothingLoss,
    ScheduledOptim,
    Transformer,
    bleu,
    create_masks,
    get_model,
    translate_corpus_batched,
    translate_sentence,
    validate,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds across numpy, torch, and tensorflow for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)


def save_vocab(SRC: Field, TRG: Field, path: Path) -> None:
    """Save SRC and TRG vocabularies to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "src_itos": SRC.vocab.itos,
        "trg_itos": TRG.vocab.itos,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(SRC: Field, TRG: Field, path: Path) -> None:
    """Load SRC and TRG vocabularies from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    src_itos = payload["src_itos"]
    trg_itos = payload["trg_itos"]
    SRC.vocab = Vocab(
        stoi={tok: idx for idx, tok in enumerate(src_itos)},
        itos=list(src_itos),
        freqs={},
    )
    TRG.vocab = Vocab(
        stoi={tok: idx for idx, tok in enumerate(trg_itos)},
        itos=list(trg_itos),
        freqs={},
    )


def train_transformer_case1(
    train_loader,
    valid_loader,
    valid_src: Sequence[str],
    valid_trg: Sequence[str],
    SRC: Field,
    TRG: Field,
    device: torch.device,
    checkpoint_path: Path,
    d_model: int = 512,
    n_layers: int = 6,
    heads: int = 8,
    dropout: int | float = 0.1,
    d_ff: int = 2048,
    epochs: int = 10,
    n_warmup_steps: int = 4000,
    lr_mul: float = 2.0,
) -> Tuple[Transformer, List[Dict[str, float]]]:
    """Train Case 1 (PyTorch Transformer) and return best model and epoch history."""
    opt_cfg = {
        "src_vocab_size": len(SRC.vocab),
        "trg_vocab_size": len(TRG.vocab),
        "d_model": d_model,
        "n_layers": n_layers,
        "heads": heads,
        "dropout": dropout,
        "d_ff": d_ff,
        "load_weights": None,
    }
    model = get_model(opt_cfg, len(SRC.vocab), len(TRG.vocab)).to(device)
    optimizer = ScheduledOptim(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr_mul=lr_mul,
        d_model=d_model,
        n_warmup_steps=n_warmup_steps,
    )
    criterion = LabelSmoothingLoss(classes=len(TRG.vocab), padding_idx=PAD_IDX, smoothing=0.1)

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    history: List[Dict[str, float]] = []
    best_bleu = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Case 1: PyTorch Transformer] Starting training on {device} "
        f"(d_model={d_model}, layers={n_layers}, heads={heads}, amp={use_amp})..."
    )

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        batch_losses: List[float] = []

        for batch in train_loader:
            src = batch.src.transpose(0, 1).to(device, non_blocking=True)
            trg = batch.trg.transpose(0, 1).to(device, non_blocking=True)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, PAD_IDX, PAD_IDX, device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(src, trg_input, src_mask, trg_mask)
                ys = trg[:, 1:].contiguous().view(-1)
                loss = criterion(preds.view(-1, preds.size(-1)), ys)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step_and_update_lr()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss = validate(model, valid_loader, criterion, PAD_IDX, PAD_IDX, device)
        val_bleu = bleu(valid_src, valid_trg, model, SRC, TRG, device, max_strlen=60, use_beam=False)
        elapsed = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "valid_loss": round(val_loss, 4),
            "valid_ppl": round(math.exp(min(val_loss, 20.0)), 3),
            "valid_bleu": round(val_bleu, 4),
            "valid_bleu_pct": round(val_bleu * 100.0, 2),
            "time_s": round(elapsed, 2),
            "lr": round(optimizer.current_lr, 6),
        }
        history.append(record)
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} (PPL {record['valid_ppl']:.2f}) | "
            f"Valid BLEU: {val_bleu * 100:.2f}% | LR: {optimizer.current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": opt_cfg,
                    "epoch": epoch,
                    "valid_bleu": val_bleu,
                },
                checkpoint_path,
            )

    # Reload best model weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history


def train_nmt_case2(
    train_loader,
    valid_loader,
    valid_src: Sequence[str],
    valid_trg: Sequence[str],
    SRC: Field,
    TRG: Field,
    device_str: str,
    checkpoint_path: Path,
    units: int = 512,
    dropout: float = 0.2,
    epochs: int = 10,
    lr: float = 1e-3,
) -> Tuple[Seq2SeqNMT, List[Dict[str, float]]]:
    """Train Case 2 (TensorFlow/Keras Seq2Seq Bi-LSTM + Scaled Luong Attention)."""
    model = Seq2SeqNMT(
        src_vocab_size=len(SRC.vocab),
        trg_vocab_size=len(TRG.vocab),
        units=units,
        dropout=dropout,
        pad_idx=PAD_IDX,
    )
    use_torch_backend = keras.backend.backend() == "torch"
    if use_torch_backend and device_str.startswith("cuda") and torch.cuda.is_available():
        model.to(device_str)

    # Warmup build pass
    dummy_src = torch.ones((2, 5), dtype=torch.long, device=device_str)
    dummy_trg = torch.ones((2, 4), dtype=torch.long, device=device_str)
    _ = model((dummy_src, dummy_trg), training=False)

    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=5.0)
    use_amp = use_torch_backend and device_str.startswith("cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    history: List[Dict[str, float]] = []
    best_bleu = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[Case 2: TensorFlow/Keras Seq2Seq Bi-LSTM + Scaled Luong Attention] Starting training on {device_str} "
        f"(units={units}, dropout={dropout}, backend={keras.backend.backend()}, amp={use_amp})..."
    )

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        # Learning rate decay schedule starting at epoch 7 (as in tensorflow/nmt IWSLT15)
        if epoch >= 7:
            new_lr = lr * (0.5 ** (epoch - 6))
            optimizer.learning_rate.assign(new_lr)
        curr_lr = float(keras.ops.convert_to_numpy(optimizer.learning_rate))

        batch_losses: List[float] = []
        for batch in train_loader:
            src = batch.src.transpose(0, 1).to(device_str, non_blocking=True)
            trg = batch.trg.transpose(0, 1).to(device_str, non_blocking=True)
            trg_in = trg[:, :-1]
            trg_out = trg[:, 1:]

            model.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model((src, trg_in), training=True)
                loss = masked_nmt_loss(trg_out, logits, pad_idx=PAD_IDX)

            loss.backward()
            trainable_weights = model.trainable_weights
            grads = [w.value.grad for w in trainable_weights]
            optimizer.apply(grads, trainable_weights)
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_nmt_loss(model, valid_loader, pad_idx=PAD_IDX, device=device_str)
        val_preds = translate_corpus_nmt_batched(
            valid_src, model, SRC, TRG, device=device_str, batch_size=64, max_len=60
        )
        val_bleu = bleu_score(
            [TRG.preprocess(s) for s in val_preds],
            [[TRG.preprocess(s)] for s in valid_trg],
        )
        elapsed = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "valid_loss": round(val_loss, 4),
            "valid_ppl": round(math.exp(min(val_loss, 20.0)), 3),
            "valid_bleu": round(val_bleu, 4),
            "valid_bleu_pct": round(val_bleu * 100.0, 2),
            "time_s": round(elapsed, 2),
            "lr": round(curr_lr, 6),
        }
        history.append(record)
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} (PPL {record['valid_ppl']:.2f}) | "
            f"Valid BLEU: {val_bleu * 100:.2f}% | LR: {curr_lr:.6f} | Time: {elapsed:.1f}s"
        )

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            model.save_weights(str(checkpoint_path))

    model.load_weights(str(checkpoint_path))
    return model, history


def compute_length_bucket_bleu(
    src_sentences: Sequence[str],
    ref_sentences: Sequence[str],
    pred_sentences: Sequence[str],
    SRC: Field,
    TRG: Field,
) -> List[Dict[str, Any]]:
    """Compute BLEU score across source sentence length buckets."""
    buckets = [
        ("< 10", 0, 10),
        ("10 - 19", 10, 20),
        ("20 - 29", 20, 30),
        ("30 - 39", 30, 40),
        (">= 40", 40, 1000),
    ]
    results: List[Dict[str, Any]] = []
    src_lengths = [len(SRC.preprocess(s)) for s in src_sentences]

    for label, low, high in buckets:
        indices = [i for i, length in enumerate(src_lengths) if low <= length < high]
        if not indices:
            continue
        sub_preds = [TRG.preprocess(pred_sentences[i]) for i in indices]
        sub_refs = [[TRG.preprocess(ref_sentences[i])] for i in indices]
        score = bleu_score(sub_preds, sub_refs) * 100.0
        results.append(
            {
                "bucket": label,
                "count": len(indices),
                "bleu_pct": round(score, 2),
            }
        )
    return results


def plot_training_curves(
    tf_hist: List[Dict[str, float]],
    nmt_hist: List[Dict[str, float]],
    out_path: Path,
) -> None:
    """Plot side-by-side training loss, validation loss, and validation BLEU curves."""
    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=160)

    epochs_tf = [r["epoch"] for r in tf_hist]
    epochs_nmt = [r["epoch"] for r in nmt_hist]

    # Panel 1: Training & Validation Loss
    axes[0].plot(
        epochs_tf,
        [r["train_loss"] for r in tf_hist],
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        label="Case 1: Transformer (Train)",
    )
    axes[0].plot(
        epochs_tf,
        [r["valid_loss"] for r in tf_hist],
        "s--",
        color="#1f77b4",
        linewidth=2.0,
        label="Case 1: Transformer (Valid - tst2012)",
    )
    axes[0].plot(
        epochs_nmt,
        [r["train_loss"] for r in nmt_hist],
        "o-",
        color="#ff7f0e",
        linewidth=2.2,
        label="Case 2: NMT Bi-LSTM+Luong (Train)",
    )
    axes[0].plot(
        epochs_nmt,
        [r["valid_loss"] for r in nmt_hist],
        "s--",
        color="#ff7f0e",
        linewidth=2.0,
        label="Case 2: NMT Bi-LSTM+Luong (Valid - tst2012)",
    )
    axes[0].set_title("Training & Validation Loss per Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(fontsize=9)

    # Panel 2: Validation Perplexity
    axes[1].plot(
        epochs_tf,
        [r["valid_ppl"] for r in tf_hist],
        "o-",
        color="#1f77b4",
        linewidth=2.4,
        label="Case 1: PyTorch Transformer",
    )
    axes[1].plot(
        epochs_nmt,
        [r["valid_ppl"] for r in nmt_hist],
        "s-",
        color="#ff7f0e",
        linewidth=2.4,
        label="Case 2: TF/Keras NMT (Bi-LSTM + Luong)",
    )
    axes[1].set_title("Validation Perplexity on tst2012 (Lower is Better)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity (PPL)")
    axes[1].legend(fontsize=10)

    # Panel 3: Validation BLEU (%)
    axes[2].plot(
        epochs_tf,
        [r["valid_bleu_pct"] for r in tf_hist],
        "o-",
        color="#2ca02c",
        linewidth=2.5,
        label="Case 1: PyTorch Transformer",
    )
    axes[2].plot(
        epochs_nmt,
        [r["valid_bleu_pct"] for r in nmt_hist],
        "s-",
        color="#d62728",
        linewidth=2.5,
        label="Case 2: TF/Keras NMT (Bi-LSTM + Luong)",
    )
    axes[2].set_title("Validation BLEU (%) on tst2012 (Higher is Better)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("BLEU Score (%)")
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_and_buckets(
    tf_metrics_greedy: Dict[str, float],
    tf_metrics_beam: Dict[str, float],
    nmt_metrics_greedy: Dict[str, float],
    nmt_metrics_beam: Dict[str, float],
    tf_buckets: List[Dict[str, Any]],
    nmt_buckets: List[Dict[str, Any]],
    out_metrics_path: Path,
    out_bucket_path: Path,
) -> None:
    """Plot test-set BLEU comparison bars and sentence-length bucket curves."""
    sns.set_theme(style="whitegrid")

    # 1. Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    metric_keys = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "bleu_percent", "sacrebleu"]
    metric_labels = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "Corpus BLEU (%)", "SacreBLEU"]

    x = np.arange(len(metric_keys))
    width = 0.20

    bars1 = ax.bar(
        x - 1.5 * width,
        [tf_metrics_greedy[k] for k in metric_keys],
        width,
        label="Case 1: Transformer (Greedy)",
        color="#4c72b0",
    )
    bars2 = ax.bar(
        x - 0.5 * width,
        [tf_metrics_beam[k] for k in metric_keys],
        width,
        label="Case 1: Transformer (Beam k=5)",
        color="#1f4e78",
    )
    bars3 = ax.bar(
        x + 0.5 * width,
        [nmt_metrics_greedy[k] for k in metric_keys],
        width,
        label="Case 2: NMT Bi-LSTM+Luong (Greedy)",
        color="#dd8452",
    )
    bars4 = ax.bar(
        x + 1.5 * width,
        [nmt_metrics_beam[k] for k in metric_keys],
        width,
        label="Case 2: NMT Bi-LSTM+Luong (Beam k=5)",
        color="#b8541a",
    )

    for bars in (bars1, bars2, bars3, bars4):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    ax.set_title("IWSLT 2015 En-Vi Test Set (tst2013) BLEU Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, max(tf_metrics_beam["bleu_1"], nmt_metrics_beam["bleu_1"]) * 1.15)
    ax.legend(fontsize=9.5, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_metrics_path, bbox_inches="tight")
    plt.close(fig)

    # 2. Length Bucket BLEU Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    labels = [b["bucket"] for b in tf_buckets]
    counts = [b["count"] for b in tf_buckets]
    x_b = np.arange(len(labels))
    w_b = 0.35

    b_tf = ax.bar(
        x_b - w_b / 2,
        [b["bleu_pct"] for b in tf_buckets],
        w_b,
        label="Case 1: PyTorch Transformer",
        color="#2b5c8f",
    )
    b_nmt = ax.bar(
        x_b + w_b / 2,
        [b["bleu_pct"] for b in nmt_buckets],
        w_b,
        label="Case 2: TF/Keras NMT (Bi-LSTM + Luong)",
        color="#d95f02",
    )

    for bars in (b_tf, b_nmt):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_title("BLEU (%) by Source Sentence Length on tst2013", fontsize=13, fontweight="bold")
    ax.set_xticks(x_b)
    ax.set_xticklabels([f"{lbl}\n(n={cnt})" for lbl, cnt in zip(labels, counts)], fontsize=10)
    ax.set_xlabel("Source English Sentence Length (Tokens)")
    ax.set_ylabel("BLEU Score (%)")
    ax.set_ylim(0, max(max(b["bleu_pct"] for b in tf_buckets), max(b["bleu_pct"] for b in nmt_buckets)) * 1.18)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_bucket_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_transformer_attention_heatmap(
    sentence: str,
    model: Transformer,
    SRC: Field,
    TRG: Field,
    device: torch.device,
    out_path: Path,
) -> str:
    """Translate `sentence` with Case 1 Transformer and plot Multi-Head Cross-Attention heatmaps."""
    model.eval()
    translation = translate_sentence(sentence, model, SRC, TRG, device, k=5, max_len=60)
    src_tokens = SRC.preprocess(sentence)
    trg_tokens = [ "<sos>" ] + TRG.preprocess(translation)

    src_ids = torch.tensor([[SRC.vocab.stoi.get(t, UNK_IDX) for t in src_tokens]], dtype=torch.long, device=device)
    trg_ids = torch.tensor([[TRG.vocab.stoi.get(t, UNK_IDX) for t in trg_tokens]], dtype=torch.long, device=device)
    src_mask, trg_mask = create_masks(src_ids, trg_ids, PAD_IDX, PAD_IDX, device)
    _ = model(src_ids, trg_ids, src_mask, trg_mask)

    # Extract last decoder layer cross-attention scores: shape (1, heads, trg_len, src_len)
    cross_attn = model.decoder.layers[-1].attn_2.viz_scores[0].detach().cpu().numpy()
    pred_display_tokens = TRG.preprocess(translation) + ["<eos>"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=160)
    for head_idx, ax in enumerate(axes.flat):
        mat = cross_attn[head_idx, : len(pred_display_tokens), : len(src_tokens)]
        sns.heatmap(
            mat,
            xticklabels=src_tokens,
            yticklabels=pred_display_tokens,
            cmap="YlGnBu",
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"Decoder Layer {model.decoder.N} - Head {head_idx + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Source (English)")
        ax.set_ylabel("Target (Vietnamese)")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        f"Case 1: PyTorch Transformer Multi-Head Cross-Attention\nEN: {sentence}\nVI: {translation}",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return translation


def plot_nmt_attention_heatmap(
    sentence: str,
    model: Seq2SeqNMT,
    SRC: Field,
    TRG: Field,
    device_str: str,
    out_path: Path,
) -> str:
    """Translate `sentence` with Case 2 Seq2SeqNMT and plot Scaled Luong Attention heatmap."""
    translation, attn_matrix, src_tokens, pred_words = translate_sentence_nmt(
        sentence, model, SRC, TRG, device=device_str, beam_width=5, max_len=60
    )
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=160)
    sns.heatmap(
        attn_matrix,
        xticklabels=src_tokens,
        yticklabels=pred_words,
        cmap="magma",
        annot=True if (len(src_tokens) <= 14 and len(pred_words) <= 14) else False,
        fmt=".2f",
        ax=ax,
        cbar=True,
    )
    ax.set_title(
        f"Case 2: TensorFlow/Keras Seq2Seq Scaled Luong Attention\nEN: {sentence}\nVI: {translation}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Source Tokens (English)")
    ax.set_ylabel("Predicted Tokens (Vietnamese)")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return translation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare Case 1 (Transformer) and Case 2 (NMT Seq2Seq).")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs-transformer", type=int, default=10)
    parser.add_argument("--epochs-nmt", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-strlen", type=int, default=50)
    parser.add_argument("--beam-eval-count", type=int, default=1268)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    paths = ensure_iwslt15_data(".")
    checkpoints_dir = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading IWSLT 2015 English-Vietnamese Parallel Corpus...")
    raw_train_en, raw_train_vi = read_parallel_data(paths["train_en"], paths["train_vi"])
    train_en, train_vi = read_parallel_data(
        paths["train_en"], paths["train_vi"], max_strlen=args.max_strlen
    )
    dev_en, dev_vi = read_parallel_data(paths["dev_en"], paths["dev_vi"])
    test_en, test_vi = read_parallel_data(paths["test_en"], paths["test_vi"])
    print(
        f"   Raw pairs -> Train: {len(raw_train_en):,}, "
        f"Dev (tst2012): {len(dev_en):,}, Test (tst2013): {len(test_en):,}"
    )

    SRC = Field(lang="en", lower=True)
    TRG = Field(lang="vi", lower=True, init_token="<sos>", eos_token="<eos>")

    SRC.build_vocab(train_en, min_freq=3, max_size=17200)
    TRG.build_vocab(train_vi, min_freq=3, max_size=8000)
    save_vocab(SRC, TRG, checkpoints_dir / "vocab.json")

    train_dataset = ParallelTranslationDataset(
        train_en, train_vi, SRC, TRG, max_len=args.max_strlen
    )
    dev_dataset = ParallelTranslationDataset(dev_en, dev_vi, SRC, TRG, max_len=80)
    test_dataset = ParallelTranslationDataset(test_en, test_vi, SRC, TRG, max_len=80)

    print(
        f"   Filtered Train (<={args.max_strlen} tokens): {len(train_dataset):,} pairs | "
        f"Vocab EN: {len(SRC.vocab):,} tokens | Vocab VI: {len(TRG.vocab):,} tokens"
    )

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # Train Case 1: PyTorch Transformer
    # -------------------------------------------------------------------------
    tf_ckpt = checkpoints_dir / "transformer_best.pt"
    tf_model, tf_history = train_transformer_case1(
        train_loader=train_loader,
        valid_loader=dev_loader,
        valid_src=dev_en,
        valid_trg=dev_vi,
        SRC=SRC,
        TRG=TRG,
        device=device,
        checkpoint_path=tf_ckpt,
        d_model=512,
        n_layers=6,
        heads=8,
        dropout=0.1,
        d_ff=2048,
        epochs=args.epochs_transformer,
        n_warmup_steps=4000,
        lr_mul=2.0,
    )

    # -------------------------------------------------------------------------
    # Train Case 2: TensorFlow/Keras Seq2Seq Bi-LSTM + Scaled Luong Attention
    # -------------------------------------------------------------------------
    nmt_ckpt = checkpoints_dir / "nmt_seq2seq_best.weights.h5"
    nmt_model, nmt_history = train_nmt_case2(
        train_loader=train_loader,
        valid_loader=dev_loader,
        valid_src=dev_en,
        valid_trg=dev_vi,
        SRC=SRC,
        TRG=TRG,
        device_str=device_str,
        checkpoint_path=nmt_ckpt,
        units=512,
        dropout=0.2,
        epochs=args.epochs_nmt,
        lr=1e-3,
    )

    # -------------------------------------------------------------------------
    # Evaluate Both Models on Test Set (tst2013 - 1,268 sentences)
    # -------------------------------------------------------------------------
    print("\n2. Evaluating both models on Test Set (tst2013 - 1,268 parallel sentences)...")
    tf_criterion = LabelSmoothingLoss(classes=len(TRG.vocab), padding_idx=PAD_IDX, smoothing=0.0)
    tf_test_loss = validate(tf_model, test_loader, tf_criterion, PAD_IDX, PAD_IDX, device)
    nmt_test_loss = evaluate_nmt_loss(nmt_model, test_loader, pad_idx=PAD_IDX, device=device_str)

    # Greedy Decoding on full tst2013
    t0 = time.perf_counter()
    tf_preds_greedy = translate_corpus_batched(test_en, tf_model, SRC, TRG, device, batch_size=64, max_len=60)
    tf_greedy_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    nmt_preds_greedy = translate_corpus_nmt_batched(
        test_en, nmt_model, SRC, TRG, device=device_str, batch_size=64, max_len=60
    )
    nmt_greedy_time = time.perf_counter() - t0

    # Beam Search (k=5) evaluation on tst2013
    beam_n = min(args.beam_eval_count, len(test_en))
    print(f"   Running Beam Search (k=5) on {beam_n} test sentences...")
    t0 = time.perf_counter()
    tf_preds_beam = [
        translate_sentence(s, tf_model, SRC, TRG, device, k=5, max_len=60) for s in test_en[:beam_n]
    ]
    tf_beam_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    nmt_preds_beam = [
        translate_sentence_nmt(s, nmt_model, SRC, TRG, device=device_str, beam_width=5, max_len=60)[0]
        for s in test_en[:beam_n]
    ]
    nmt_beam_time = time.perf_counter() - t0

    tf_metrics_greedy = compute_translation_metrics(tf_preds_greedy, test_vi, TRG.preprocess)
    nmt_metrics_greedy = compute_translation_metrics(nmt_preds_greedy, test_vi, TRG.preprocess)
    tf_metrics_beam = compute_translation_metrics(tf_preds_beam, test_vi[:beam_n], TRG.preprocess)
    nmt_metrics_beam = compute_translation_metrics(nmt_preds_beam, test_vi[:beam_n], TRG.preprocess)

    tf_buckets = compute_length_bucket_bleu(test_en[:beam_n], test_vi[:beam_n], tf_preds_beam, SRC, TRG)
    nmt_buckets = compute_length_bucket_bleu(test_en[:beam_n], test_vi[:beam_n], nmt_preds_beam, SRC, TRG)

    tf_params = sum(p.numel() for p in tf_model.parameters() if p.requires_grad)
    nmt_params = int(sum( int( np.prod(w.shape) ) for w in nmt_model.trainable_weights ))

    print("\n================ TEST SET (tst2013) SUMMARY ================")
    print(
        f"Case 1 (PyTorch Transformer)     | Params: {tf_params:,} | "
        f"Greedy BLEU: {tf_metrics_greedy['bleu_percent']:.2f}% (SacreBLEU {tf_metrics_greedy['sacrebleu']:.2f}) | "
        f"Beam(k=5) BLEU: {tf_metrics_beam['bleu_percent']:.2f}% (SacreBLEU {tf_metrics_beam['sacrebleu']:.2f})"
    )
    print(
        f"Case 2 (TF/Keras Seq2Seq Luong)  | Params: {nmt_params:,} | "
        f"Greedy BLEU: {nmt_metrics_greedy['bleu_percent']:.2f}% (SacreBLEU {nmt_metrics_greedy['sacrebleu']:.2f}) | "
        f"Beam(k=5) BLEU: {nmt_metrics_beam['bleu_percent']:.2f}% (SacreBLEU {nmt_metrics_beam['sacrebleu']:.2f})"
    )
    print("============================================================\n")

    # -------------------------------------------------------------------------
    # Generate Plots & Sample Translations
    # -------------------------------------------------------------------------
    plot_training_curves(tf_history, nmt_history, results_dir / "training_curves.png")
    plot_metrics_and_buckets(
        tf_metrics_greedy=tf_metrics_greedy,
        tf_metrics_beam=tf_metrics_beam,
        nmt_metrics_greedy=nmt_metrics_greedy,
        nmt_metrics_beam=nmt_metrics_beam,
        tf_buckets=tf_buckets,
        nmt_buckets=nmt_buckets,
        out_metrics_path=results_dir / "metrics_comparison.png",
        out_bucket_path=results_dir / "length_bucket_bleu.png",
    )

    viz_sentence = "When I was little , I thought my country was the best on the planet , and I grew up singing a song called &quot; Nothing To Envy . &quot;"
    if viz_sentence not in test_en:
        viz_sentence = test_en[0]
    plot_transformer_attention_heatmap(
        viz_sentence, tf_model, SRC, TRG, device, results_dir / "transformer_attention_heatmap.png"
    )
    plot_nmt_attention_heatmap(
        viz_sentence, nmt_model, SRC, TRG, device_str, results_dir / "nmt_luong_attention_heatmap.png"
    )

    sample_indices = [0, 1, 2, 5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 300, 400]
    sample_translations = []
    for idx in sample_indices:
        if idx < len(test_en):
            sample_translations.append(
                {
                    "index": idx,
                    "source_en": test_en[idx],
                    "reference_vi": detokenize_sentence(test_vi[idx]),
                    "transformer_greedy_vi": tf_preds_greedy[idx],
                    "transformer_beam_vi": tf_preds_beam[idx] if idx < len(tf_preds_beam) else tf_preds_greedy[idx],
                    "nmt_greedy_vi": nmt_preds_greedy[idx],
                    "nmt_beam_vi": nmt_preds_beam[idx] if idx < len(nmt_preds_beam) else nmt_preds_greedy[idx],
                }
            )

    (results_dir / "sample_translations.json").write_text(
        json.dumps(sample_translations, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    comparison_summary = {
        "dataset": {
            "name": "IWSLT 2015 English-Vietnamese Parallel Corpus",
            "train_raw_pairs": len(train_en),
            "train_filtered_pairs": len(train_dataset),
            "dev_pairs_tst2012": len(dev_en),
            "test_pairs_tst2013": len(test_en),
            "src_vocab_size": len(SRC.vocab),
            "trg_vocab_size": len(TRG.vocab),
            "max_strlen": args.max_strlen,
        },
        "case1_transformer_pytorch": {
            "architecture": "Transformer (6 Encoder + 6 Decoder layers, d_model=512, heads=8, d_ff=2048, Pre-LN, Sinusoidal PE)",
            "optimizer": "Noam ScheduledOptim (Adam betas=(0.9, 0.98), warmup=4000, lr_mul=2.0) + LabelSmoothing(0.1)",
            "parameters": tf_params,
            "avg_epoch_time_s": round(float(np.mean([r["time_s"] for r in tf_history])), 2),
            "total_train_time_s": round(float(np.sum([r["time_s"] for r in tf_history])), 2),
            "test_loss": round(tf_test_loss, 4),
            "test_ppl": round(math.exp(min(tf_test_loss, 20.0)), 3),
            "greedy_decode_time_s": round(tf_greedy_time, 2),
            "greedy_ms_per_sentence": round(tf_greedy_time * 1000.0 / len(test_en), 2),
            "beam_decode_time_s": round(tf_beam_time, 2),
            "beam_ms_per_sentence": round(tf_beam_time * 1000.0 / beam_n, 2),
            "metrics_greedy": tf_metrics_greedy,
            "metrics_beam_k5": tf_metrics_beam,
            "length_buckets": tf_buckets,
            "history": tf_history,
        },
        "case2_nmt_tensorflow": {
            "architecture": "Seq2Seq 2-Layer LSTM (Bi-LSTM Encoder + 2-Layer Decoder + Scaled Luong Attention, units=512, dropout=0.2)",
            "optimizer": "Adam (lr=1e-3, clipnorm=5.0, decay=0.5x after epoch 6)",
            "parameters": nmt_params,
            "avg_epoch_time_s": round(float(np.mean([r["time_s"] for r in nmt_history])), 2),
            "total_train_time_s": round(float(np.sum([r["time_s"] for r in nmt_history])), 2),
            "test_loss": round(nmt_test_loss, 4),
            "test_ppl": round(math.exp(min(nmt_test_loss, 20.0)), 3),
            "greedy_decode_time_s": round(nmt_greedy_time, 2),
            "greedy_ms_per_sentence": round(nmt_greedy_time * 1000.0 / len(test_en), 2),
            "beam_decode_time_s": round(nmt_beam_time, 2),
            "beam_ms_per_sentence": round(nmt_beam_time * 1000.0 / beam_n, 2),
            "metrics_greedy": nmt_metrics_greedy,
            "metrics_beam_k5": nmt_metrics_beam,
            "length_buckets": nmt_buckets,
            "history": nmt_history,
        },
    }

    (results_dir / "comparison_results.json").write_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"3. Saved all metrics, plots, and sample translations to {results_dir.resolve()}/")


if __name__ == "__main__":
    main()
