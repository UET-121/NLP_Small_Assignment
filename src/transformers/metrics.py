"""Evaluation metrics for English-Vietnamese Neural Machine Translation (BLEU, SacreBLEU, Perplexity)."""

from __future__ import annotations

from collections import Counter
import math
from typing import Dict, List, Sequence

import sacrebleu


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def bleu_score(
    candidate_corpus: Sequence[Sequence[str]],
    references_corpus: Sequence[Sequence[Sequence[str]]],
    max_n: int = 4,
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """Drop-in replacement for `torchtext.data.metrics.bleu_score` (returns score in [0.0, 1.0])."""
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    candidate_len = 0
    refs_len = 0

    for candidate, refs in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)
        ref_lens = [len(r) for r in refs]
        refs_len += min(ref_lens, key=lambda rl: (abs(rl - len(candidate)), rl))

        for n in range(1, max_n + 1):
            c_counts = _ngram_counts(candidate, n)
            total_counts[n - 1] += sum(c_counts.values())

            max_ref_counts: Counter = Counter()
            for r in refs:
                r_counts = _ngram_counts(r, n)
                for ngram, cnt in r_counts.items():
                    if cnt > max_ref_counts[ngram]:
                        max_ref_counts[ngram] = cnt

            for ngram, cnt in c_counts.items():
                clipped_counts[n - 1] += min(cnt, max_ref_counts[ngram])

    if candidate_len == 0 or any(c == 0 for c in clipped_counts):
        return 0.0

    precisions = [clipped_counts[i] / total_counts[i] for i in range(max_n)]
    log_precision_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
    bp = 1.0 if candidate_len > refs_len else math.exp(1.0 - (refs_len / candidate_len))
    return float(bp * math.exp(log_precision_sum))


def compute_translation_metrics(
    pred_sentences: Sequence[str],
    ref_sentences: Sequence[str],
    tokenize_fn=None,
) -> Dict[str, float]:
    """Compute comprehensive BLEU metrics (torchtext-style [0,1], NMT % scale, BLEU-1..4, SacreBLEU)."""
    if tokenize_fn is not None:
        cand_toks: List[List[str]] = [tokenize_fn(s) for s in pred_sentences]
        ref_toks: List[List[List[str]]] = [[tokenize_fn(r)] for r in ref_sentences]
    else:
        cand_toks = [s.lower().strip().split() for s in pred_sentences]
        ref_toks = [[r.lower().strip().split()] for r in ref_sentences]

    b4 = bleu_score(cand_toks, ref_toks, max_n=4, weights=(0.25, 0.25, 0.25, 0.25))
    b1 = bleu_score(cand_toks, ref_toks, max_n=1, weights=(1.0,))
    b2 = bleu_score(cand_toks, ref_toks, max_n=2, weights=(0.5, 0.5))
    b3 = bleu_score(cand_toks, ref_toks, max_n=3, weights=(1 / 3, 1 / 3, 1 / 3))

    # Normalized strings for sacrebleu
    norm_preds = [" ".join(toks) for toks in cand_toks]
    norm_refs = [[" ".join(refs[0]) for refs in ref_toks]]
    sb = sacrebleu.corpus_bleu(norm_preds, norm_refs, tokenize="none")

    return {
        "bleu": round(b4, 4),
        "bleu_percent": round(b4 * 100.0, 2),
        "sacrebleu": round(float(sb.score), 2),
        "bleu_1": round(b1 * 100.0, 2),
        "bleu_2": round(b2 * 100.0, 2),
        "bleu_3": round(b3 * 100.0, 2),
        "bleu_4": round(b4 * 100.0, 2),
    }
