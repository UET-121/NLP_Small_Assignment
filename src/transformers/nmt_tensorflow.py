"""Case 2: TensorFlow/Keras 2.x+ Reimplementation of `tensorflow/nmt` (Seq2Seq Bi-LSTM + Scaled Luong Attention).

Reference:
    Minh-Thang Luong, Eugene Brevdo, Rui Zhao (2017),
    "Neural Machine Translation (seq2seq) Tutorial", https://github.com/tensorflow/nmt
    IWSLT 2015 English-Vietnamese benchmark configuration:
      - 2-layer LSTM (Bi-LSTM encoder + 2-layer LSTM decoder), num_units=512, dropout=0.2
      - Scaled Luong multiplicative attention: score(h_t, h_s) = (h_t^T W_a h_s) / sqrt(d)
      - Attentional hidden state: h~_t = tanh(W_c [c_t ; h_t])
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, ops

from transformers.data_utils import (
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    UNK_IDX,
    Field,
    detokenize_sentence,
)
from transformers.metrics import bleu_score


class ScaledLuongAttention(layers.Layer):
    """Scaled Multiplicative Luong Attention as in `tensorflow/nmt` (`--attention=scaled_luong`)."""

    def __init__(self, units: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.scale = 1.0 / math.sqrt(float(units))
        self.memory_dense = layers.Dense(units, use_bias=False, name="memory_layer")
        self.attention_layer = layers.Dense(units, activation="tanh", use_bias=False, name="attention_layer")
        self.dropout = layers.Dropout(dropout)

    def call(
        self,
        query: tf.Tensor,
        memory: tf.Tensor,
        src_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute Luong attentional vector and alignment weights.

        Args:
            query: Decoder hidden states (batch, trg_len, units)
            memory: Encoder outputs (batch, src_len, units)
            src_mask: Boolean/float mask (batch, src_len) where 1/True = valid token, 0/False = pad

        Returns:
            attentional_hidden: (batch, trg_len, units) = tanh(W_c [context ; query])
            alignments: (batch, trg_len, src_len) attention weights
        """
        # keys: (batch, src_len, units)
        keys = self.memory_dense(memory)
        # scores: (batch, trg_len, src_len)
        scores = ops.matmul(query, ops.transpose(keys, axes=(0, 2, 1))) * self.scale

        if src_mask is not None:
            # Expand mask to (batch, 1, src_len)
            mask_float = ops.cast( ops.expand_dims(src_mask, axis=1), scores.dtype)
            scores = scores * mask_float + (1.0 - mask_float) * (-1e9)

        alignments = ops.softmax(scores, axis=-1)
        alignments_drop = self.dropout(alignments, training=training)

        # context: (batch, trg_len, units)
        context = ops.matmul(alignments_drop, memory)

        # Concatenate [context ; query] and pass through tanh(W_c [c_t ; h_t])
        combined = ops.concatenate([context, query], axis=-1)
        attentional_hidden = self.attention_layer(combined)
        return attentional_hidden, alignments


class NMTEncoder(layers.Layer):
    """2-layer Bi-LSTM Encoder following `tensorflow/nmt` (`--encoder_type=bi --num_layers=2`)."""

    def __init__(self, vocab_size: int, units: int = 512, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.units = units
        self.embedding = layers.Embedding(vocab_size, units, name="src_embedding")
        self.dropout = layers.Dropout(dropout)

        # Layer 1: Bidirectional LSTM with units // 2 per direction -> concatenated size = units
        half_units = units // 2
        self.bi_lstm = layers.Bidirectional(
            layers.LSTM(half_units, return_sequences=True, return_state=True),
            merge_mode="concat",
            name="encoder_bi_lstm",
        )
        # Layer 2: Unidirectional LSTM of size `units`
        self.uni_lstm = layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name="encoder_lstm_2",
        )

    def call(
        self,
        src_tokens: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, List[Tuple[tf.Tensor, tf.Tensor]]]:
        x = self.embedding(src_tokens)
        x = self.dropout(x, training=training)

        bi_out, fw_h, fw_c, bw_h, bw_c = self.bi_lstm(x, training=training)
        h1 = ops.concatenate([fw_h, bw_h], axis=-1)
        c1 = ops.concatenate([fw_c, bw_c], axis=-1)

        bi_out_drop = self.dropout(bi_out, training=training)
        uni_out, h2, c2 = self.uni_lstm(bi_out_drop, training=training)
        # Residual connection on second encoder layer
        encoder_outputs = uni_out + bi_out
        encoder_states = [(h1, c1), (h2, c2)]
        return encoder_outputs, encoder_states


class NMTDecoder(layers.Layer):
    """2-layer LSTM Decoder with Scaled Luong Attention and Output Projection."""

    def __init__(self, vocab_size: int, units: int = 512, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.units = units
        self.embedding = layers.Embedding(vocab_size, units, name="trg_embedding")
        self.dropout = layers.Dropout(dropout)

        self.lstm_1 = layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name="decoder_lstm_1",
        )
        self.lstm_2 = layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
            name="decoder_lstm_2",
        )
        self.attention = ScaledLuongAttention(units, dropout=dropout, name="scaled_luong_attention")
        self.out_proj = layers.Dense(vocab_size, use_bias=False, name="output_projection")

    def call(
        self,
        trg_tokens: tf.Tensor,
        encoder_outputs: tf.Tensor,
        initial_states: List[Tuple[tf.Tensor, tf.Tensor]],
        src_mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, List[Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
        x = self.embedding(trg_tokens)
        x = self.dropout(x, training=training)

        out_1, h1, c1 = self.lstm_1(x, initial_state=initial_states[0], training=training)
        out_1_drop = self.dropout(out_1, training=training)

        out_2, h2, c2 = self.lstm_2(out_1_drop, initial_state=initial_states[1], training=training)
        # Residual connection between decoder LSTM layers
        dec_hidden = self.dropout(out_2 + out_1, training=training)

        attn_hidden, alignments = self.attention(
            dec_hidden,
            encoder_outputs,
            src_mask=src_mask,
            training=training,
        )
        attn_hidden = self.dropout(attn_hidden, training=training)
        logits = self.out_proj(attn_hidden)
        new_states = [(h1, c1), (h2, c2)]
        return logits, new_states, alignments


class Seq2SeqNMT(keras.Model):
    """Complete `tensorflow/nmt` Seq2Seq Model with Bi-LSTM Encoder & Scaled Luong Attention Decoder."""

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        units: int = 512,
        dropout: float = 0.2,
        pad_idx: int = PAD_IDX,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.units = units
        self.dropout_rate = dropout
        self.pad_idx = pad_idx

        self.encoder = NMTEncoder(src_vocab_size, units=units, dropout=dropout)
        self.decoder = NMTDecoder(trg_vocab_size, units=units, dropout=dropout)
        self.last_attention_weights: Optional[tf.Tensor] = None

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        src_tokens, trg_input = inputs
        src_mask = ops.not_equal(src_tokens, self.pad_idx)
        encoder_outputs, encoder_states = self.encoder(src_tokens, training=training)
        logits, _, alignments = self.decoder(
            trg_input,
            encoder_outputs,
            encoder_states,
            src_mask=src_mask,
            training=training,
        )
        self.last_attention_weights = alignments
        return logits


def masked_nmt_loss(y_true: tf.Tensor, y_pred_logits: tf.Tensor, pad_idx: int = PAD_IDX) -> tf.Tensor:
    """Masked cross-entropy loss ignoring `<pad>` positions."""
    loss_per_token = keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred_logits, from_logits=True
    )
    mask = ops.cast(ops.not_equal(y_true, pad_idx), loss_per_token.dtype)
    masked_loss = loss_per_token * mask
    non_pad_count = ops.maximum(ops.sum(mask), ops.cast(1.0, loss_per_token.dtype))
    return ops.sum(masked_loss) / non_pad_count


def train_step_nmt(
    model: Seq2SeqNMT,
    optimizer: keras.optimizers.Optimizer,
    src_batch,
    trg_in_batch,
    trg_out_batch,
    pad_idx: int = PAD_IDX,
) -> float:
    """Single training step for `Seq2SeqNMT` compatible with both torch and tensorflow Keras backends."""
    backend_name = keras.backend.backend()
    if backend_name == "torch":
        model.zero_grad()
        logits = model((src_batch, trg_in_batch), training=True)
        loss = masked_nmt_loss(trg_out_batch, logits, pad_idx=pad_idx)
        loss.backward()
        trainable_weights = [v for v in model.trainable_weights]
        grads = [v.value.grad for v in trainable_weights]
        optimizer.apply(grads, trainable_weights)
        return float(ops.convert_to_numpy(loss))
    else:
        with tf.GradientTape() as tape:
            logits = model((src_batch, trg_in_batch), training=True)
            loss = masked_nmt_loss(trg_out_batch, logits, pad_idx=pad_idx)
        trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return float(ops.convert_to_numpy(loss))


def evaluate_nmt_loss(
    model: Seq2SeqNMT,
    valid_loader,
    pad_idx: int = PAD_IDX,
    device: str = "cuda",
) -> float:
    """Evaluate average validation loss of `Seq2SeqNMT`."""
    losses: List[float] = []
    backend_name = keras.backend.backend()
    for batch in valid_loader:
        src = batch.src_batch
        trg = batch.trg_batch
        if backend_name == "torch":
            src = src.to(device)
            trg = trg.to(device)
            with torch.no_grad():
                logits = model((src, trg[:, :-1]), training=False)
                loss = masked_nmt_loss(trg[:, 1:], logits, pad_idx=pad_idx)
                losses.append(float(ops.convert_to_numpy(loss)))
        else:
            src_np = src.numpy()
            trg_np = trg.numpy()
            logits = model((src_np, trg_np[:, :-1]), training=False)
            loss = masked_nmt_loss(trg_np[:, 1:], logits, pad_idx=pad_idx)
            losses.append(float(ops.convert_to_numpy(loss)))
    return float(np.mean(losses)) if losses else 0.0


def greedy_decode_nmt_batch(
    src_tokens,
    model: Seq2SeqNMT,
    pad_idx: int = PAD_IDX,
    sos_idx: int = SOS_IDX,
    eos_idx: int = EOS_IDX,
    max_len: int = 60,
) -> Tuple[List[List[int]], np.ndarray]:
    """Batched greedy decoding for `Seq2SeqNMT`, returning token IDs and attention matrices."""
    backend_name = keras.backend.backend()
    if backend_name == "torch" and isinstance(src_tokens, torch.Tensor):
        ctx = torch.no_grad()
    else:
        from contextlib import nullcontext

        ctx = nullcontext()

    with ctx:
        src_mask = ops.not_equal(src_tokens, pad_idx)
        encoder_outputs, states = model.encoder(src_tokens, training=False)
        bs = int(ops.shape(src_tokens)[0])

        curr_tok = ops.full((bs, 1), sos_idx, dtype="int64")
        finished = np.zeros(bs, dtype=bool)
        generated_cols: List[np.ndarray] = []
        attn_steps: List[np.ndarray] = []

        for _ in range(max_len - 1):
            logits, states, alignments = model.decoder(
                curr_tok,
                encoder_outputs,
                states,
                src_mask=src_mask,
                training=False,
            )
            next_tok = ops.argmax(logits[:, -1, :], axis=-1)
            next_tok_np = ops.convert_to_numpy(next_tok).astype(np.int64)
            generated_cols.append(next_tok_np)
            attn_steps.append(ops.convert_to_numpy(alignments[:, 0, :]))

            finished |= next_tok_np == eos_idx
            if finished.all():
                break
            curr_tok = ops.expand_dims(next_tok, axis=1)

    if generated_cols:
        gen_matrix = np.stack(generated_cols, axis=1)
        attn_matrix = np.stack(attn_steps, axis=1)
    else:
        gen_matrix = np.zeros((bs, 0), dtype=np.int64)
        attn_matrix = np.zeros((bs, 0, int(ops.shape(src_tokens)[1])), dtype=np.float32)

    results: List[List[int]] = []
    for row in gen_matrix.tolist():
        seq: List[int] = []
        for idx in row:
            if idx == eos_idx:
                break
            if idx not in (pad_idx, sos_idx):
                seq.append(idx)
        results.append(seq)

    return results, attn_matrix


def beam_search_nmt(
    src_tokens,
    model: Seq2SeqNMT,
    beam_width: int = 5,
    max_len: int = 60,
    alpha: float = 0.7,
    pad_idx: int = PAD_IDX,
    sos_idx: int = SOS_IDX,
    eos_idx: int = EOS_IDX,
) -> Tuple[List[int], np.ndarray]:
    """Single-sentence Beam Search decoding with batched beam step and length penalty for `Seq2SeqNMT`."""
    backend_name = keras.backend.backend()
    if backend_name == "torch" and isinstance(src_tokens, torch.Tensor):
        ctx = torch.no_grad()
    else:
        from contextlib import nullcontext

        ctx = nullcontext()

    with ctx:
        src_mask = ops.not_equal(src_tokens, pad_idx)
        encoder_outputs, init_states = model.encoder(src_tokens, training=False)

        # Each beam item: (log_prob, token_ids, decoder_states, attn_list, finished)
        # where each state tensor in decoder_states has shape (1, units)
        beams = [(0.0, [sos_idx], init_states, [], False)]

        for _ in range(max_len - 1):
            all_candidates = []
            active_indices = []
            for idx, (score, seq, states, attns, done) in enumerate(beams):
                if done:
                    all_candidates.append((score, seq, states, attns, True))
                else:
                    active_indices.append(idx)

            if not active_indices:
                break

            n_act = len(active_indices)
            last_toks = ops.convert_to_tensor(
                [[beams[idx][1][-1]] for idx in active_indices], dtype="int64"
            )
            enc_rep = ops.repeat(encoder_outputs, repeats=n_act, axis=0)
            mask_rep = ops.repeat(src_mask, repeats=n_act, axis=0)
            batched_states = [
                (
                    ops.concatenate([beams[idx][2][layer_i][0] for idx in active_indices], axis=0),
                    ops.concatenate([beams[idx][2][layer_i][1] for idx in active_indices], axis=0),
                )
                for layer_i in range(len(init_states))
            ]

            logits, next_states_batched, alignments = model.decoder(
                last_toks,
                enc_rep,
                batched_states,
                src_mask=mask_rep,
                training=False,
            )
            log_probs = ops.log_softmax(logits[:, -1, :], axis=-1)
            log_probs_np = ops.convert_to_numpy(log_probs)
            attn_np = ops.convert_to_numpy(alignments[:, 0, :])

            for b_pos, beam_idx in enumerate(active_indices):
                score, seq, _, attns, _ = beams[beam_idx]
                item_states = [
                    (
                        next_states_batched[layer_i][0][b_pos : b_pos + 1],
                        next_states_batched[layer_i][1][b_pos : b_pos + 1],
                    )
                    for layer_i in range(len(init_states))
                ]
                row_lp = log_probs_np[b_pos]
                row_attn = attn_np[b_pos]
                top_indices = np.argsort(row_lp)[-beam_width:][::-1]
                for tok_id in top_indices:
                    tok_int = int(tok_id)
                    new_score = score + float(row_lp[tok_int])
                    new_seq = seq + [tok_int]
                    new_attns = attns + [row_attn]
                    is_done = tok_int == eos_idx
                    all_candidates.append((new_score, new_seq, item_states, new_attns, is_done))

            # Rank by length-normalized score
            all_candidates.sort(
                key=lambda item: item[0] / ((max(1, len(item[1]) - 1)) ** alpha),
                reverse=True,
            )
            beams = all_candidates[:beam_width]

    best_score, best_seq, _, best_attns, _ = beams[0]
    cleaned: List[int] = []
    for idx in best_seq[1:]:
        if idx == eos_idx:
            break
        if idx not in (pad_idx, sos_idx):
            cleaned.append(idx)

    attn_arr = np.stack(best_attns, axis=0) if best_attns else np.zeros((0, 0), dtype=np.float32)
    return cleaned, attn_arr


def translate_sentence_nmt(
    sentence: str,
    model: Seq2SeqNMT,
    SRC: Field,
    TRG: Field,
    device: str = "cuda",
    beam_width: int = 5,
    max_len: int = 60,
) -> Tuple[str, np.ndarray, List[str], List[str]]:
    """Translate a sentence using `Seq2SeqNMT` (with beam search) and return attention matrix."""
    src_tokens = SRC.preprocess(sentence)[:max_len]
    if not src_tokens:
        return "", np.zeros((0, 0), dtype=np.float32), [], []

    indexed = [SRC.vocab.stoi.get(tok, UNK_IDX) for tok in src_tokens]
    if keras.backend.backend() == "torch":
        src_tensor = torch.tensor([indexed], dtype=torch.long, device=device)
    else:
        src_tensor = tf.constant([indexed], dtype=tf.int64)

    if beam_width > 1:
        pred_ids, attn_matrix = beam_search_nmt(
            src_tensor, model, beam_width=beam_width, max_len=max_len
        )
    else:
        batch_ids, batch_attn = greedy_decode_nmt_batch(src_tensor, model, max_len=max_len)
        pred_ids = batch_ids[0]
        attn_matrix = batch_attn[0]

    pred_words = [TRG.vocab.itos[idx] for idx in pred_ids if 0 <= idx < len(TRG.vocab)]
    translation = detokenize_sentence(" ".join(pred_words))
    return translation, attn_matrix[: len(pred_words), : len(src_tokens)], src_tokens, pred_words


def translate_corpus_nmt_batched(
    src_sentences: Sequence[str],
    model: Seq2SeqNMT,
    SRC: Field,
    TRG: Field,
    device: str = "cuda",
    batch_size: int = 64,
    max_len: int = 60,
) -> List[str]:
    """Translate a corpus of sentences in batches using `Seq2SeqNMT`."""
    predictions: List[str] = []
    use_torch = keras.backend.backend() == "torch"

    for start in range(0, len(src_sentences), batch_size):
        chunk = src_sentences[start : start + batch_size]
        encoded = [
            [SRC.vocab.stoi.get(tok, UNK_IDX) for tok in SRC.preprocess(s)[:max_len]] or [UNK_IDX]
            for s in chunk
        ]
        max_s = max(len(seq) for seq in encoded)
        arr = np.full((len(chunk), max_s), PAD_IDX, dtype=np.int64)
        for i, seq in enumerate(encoded):
            arr[i, : len(seq)] = seq

        if use_torch:
            src_tensor = torch.tensor(arr, dtype=torch.long, device=device)
        else:
            src_tensor = tf.constant(arr, dtype=tf.int64)

        pred_ids_list, _ = greedy_decode_nmt_batch(src_tensor, model, max_len=max_len)
        for ids in pred_ids_list:
            words = [TRG.vocab.itos[idx] for idx in ids if 0 <= idx < len(TRG.vocab)]
            predictions.append(detokenize_sentence(" ".join(words)))

    return predictions


def bleu_nmt(
    valid_src_data: Sequence[str],
    valid_trg_data: Sequence[str],
    model: Seq2SeqNMT,
    SRC: Field,
    TRG: Field,
    device: str = "cuda",
    max_strlen: int = 60,
) -> float:
    """Compute corpus BLEU for `Seq2SeqNMT` on validation/test set."""
    pred_sents = translate_corpus_nmt_batched(
        valid_src_data, model, SRC, TRG, device=device, batch_size=64, max_len=max_strlen
    )
    pred_toks = [TRG.preprocess(sent) for sent in pred_sents]
    trg_toks = [[TRG.preprocess(sent)] for sent in valid_trg_data]
    return bleu_score(pred_toks, trg_toks)
