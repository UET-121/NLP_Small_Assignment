"""Data loading, safe extraction, tokenization, and vocabulary utilities for IWSLT 2015 En-Vi."""

from __future__ import annotations

from collections import Counter
import html
import json
import os
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Special tokens and their fixed indices
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

GDRIVE_FILE_ID = "1Fuo_ALIFKlUvOPbK5rUA5OfAS2wKn_95"
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024  # 100 MB safety cap


def safe_extract_zip(zip_path: str, extract_to: str) -> List[str]:
    """Safely extract a zip archive after validating magic bytes, size, and member paths."""
    resolved_zip = os.path.realpath(zip_path)
    if not os.path.isfile(resolved_zip):
        raise FileNotFoundError(f"Archive not found: {resolved_zip}")

    file_size = os.path.getsize(resolved_zip)
    if file_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"Archive exceeds maximum allowed size ({MAX_ARCHIVE_BYTES} bytes).")

    with open(resolved_zip, "rb") as fp:
        magic = fp.read(4)
    if magic != b"PK\x03\x04":
        raise ValueError("Invalid ZIP archive magic bytes.")

    target_root = os.path.realpath(extract_to)
    os.makedirs(target_root, exist_ok=True)
    extracted_files: List[str] = []

    with zipfile.ZipFile(resolved_zip, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_dest = os.path.realpath(os.path.join(target_root, member.filename))
            if not member_dest.startswith(target_root + os.sep):
                raise ValueError(f"Blocked potential path traversal in zip member: {member.filename}")
            os.makedirs(os.path.dirname(member_dest), exist_ok=True)
            with zf.open(member, "r") as src_fp, open(member_dest, "wb") as dst_fp:
                dst_fp.write(src_fp.read())
            extracted_files.append(member_dest)

    return extracted_files


def ensure_iwslt15_data(work_dir: str = ".") -> Dict[str, str]:
    """Ensure IWSLT 2015 En-Vi dataset files exist in `<work_dir>/data/`."""
    base_dir = os.path.realpath(work_dir)
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    expected = {
        "train_en": os.path.join(data_dir, "train.en"),
        "train_vi": os.path.join(data_dir, "train.vi"),
        "dev_en": os.path.join(data_dir, "tst2012.en"),
        "dev_vi": os.path.join(data_dir, "tst2012.vi"),
        "test_en": os.path.join(data_dir, "tst2013.en"),
        "test_vi": os.path.join(data_dir, "tst2013.vi"),
    }

    if all(os.path.isfile(p) for p in expected.values()):
        return expected

    zip_path = os.path.join(base_dir, "en_vi.zip")
    if not os.path.isfile(zip_path):
        import gdown

        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, zip_path, quiet=False)

    safe_extract_zip(zip_path, base_dir)
    return expected


class Tokenizer:
    """Regex + punctuation-aware tokenizer compatible with demo_transformer.ipynb & IWSLT 2015."""

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._strip_re = re.compile(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]")
        self._punct_re = re.compile(r"([,\.\?\!'])")
        self._space_re = re.compile(r"\s+")

    def tokenize(self, sentence: str) -> List[str]:
        text = str(sentence)
        text = self._strip_re.sub(" ", text)
        text = self._punct_re.sub(r" \1 ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)
        text = self._space_re.sub(" ", text).strip().lower()
        if not text:
            return []
        return [tok for tok in text.split(" ") if tok]

    # Alias for compatibility with demo_transformer.ipynb `tokenize(lang).tokenizer`
    def tokenizer(self, sentence: str) -> List[str]:
        return self.tokenize(sentence)


def detokenize_sentence(text: str) -> str:
    """Clean spacing before punctuation as in demo_transformer.ipynb `multiple_replace`."""
    replacements = {" ?": "?", " !": "!", " .": ".", "' ": "'", " ,": ","}
    regex = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))
    return regex.sub(lambda mo: replacements[mo.string[mo.start() : mo.end()]], text).strip()


class Vocab:
    """Vocabulary mapping words <-> indices with `<pad>`, `<unk>`, `<sos>`, `<eos>`."""

    def __init__(self, stoi: Optional[Dict[str, int]] = None, itos: Optional[List[str]] = None):
        if stoi is not None and itos is not None:
            self.stoi = dict(stoi)
            self.itos = list(itos)
        else:
            self.itos = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
            self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    @classmethod
    def build(
        cls,
        tokenized_texts: Iterable[Sequence[str]],
        min_freq: int = 2,
        max_size: Optional[int] = 25000,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        itos = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
        sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        for word, freq in sorted_words:
            if freq < min_freq:
                break
            if word not in (PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN):
                itos.append(word)
                if max_size is not None and len(itos) >= max_size:
                    break

        stoi = {tok: idx for idx, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def encode(self, tokens: Sequence[str], add_sos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_sos:
            ids.append(SOS_IDX)
        ids.extend(self.stoi.get(t, UNK_IDX) for t in tokens)
        if add_eos:
            ids.append(EOS_IDX)
        return ids

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True, skip_special: bool = True) -> List[str]:
        words: List[str] = []
        for idx in ids:
            idx_int = int(idx)
            if stop_at_eos and idx_int == EOS_IDX:
                break
            if skip_special and idx_int in (PAD_IDX, SOS_IDX, EOS_IDX):
                continue
            if 0 <= idx_int < len(self.itos):
                words.append(self.itos[idx_int])
            else:
                words.append(UNK_TOKEN)
        return words

    def to_dict(self) -> Dict[str, List[str]]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "Vocab":
        itos = list(data["itos"])
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


class Field:
    """Drop-in replacement for legacy `torchtext.data.Field` used in demo_transformer.ipynb."""

    def __init__(
        self,
        lang: str = "en",
        init_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        lower: bool = True,
        tokenize: Optional[str] = None,
    ):
        self.lang = tokenize if isinstance(tokenize, str) else lang
        self.init_token = init_token
        self.eos_token = eos_token
        self.lower = lower
        self.tokenizer_obj = Tokenizer(self.lang)
        self.vocab: Vocab = Vocab()

    def preprocess(self, sentence: str) -> List[str]:
        return self.tokenizer_obj.tokenize(sentence)

    def build_vocab(
        self,
        sentences_or_tokens: Iterable[Sequence[str] | str],
        min_freq: int = 2,
        max_size: Optional[int] = 25000,
    ) -> None:
        tokenized = [
            self.preprocess(item) if isinstance(item, str) else list(item)
            for item in sentences_or_tokens
        ]
        self.vocab = Vocab.build(tokenized, min_freq=min_freq, max_size=max_size)


def read_parallel_data(
    src_file: str,
    trg_file: str,
    max_strlen: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Read parallel source and target files and filter empty or overly long lines."""
    with open(src_file, "r", encoding="utf-8") as f_src:
        src_lines = [line.strip() for line in f_src]
    with open(trg_file, "r", encoding="utf-8") as f_trg:
        trg_lines = [line.strip() for line in f_trg]

    if len(src_lines) != len(trg_lines):
        min_len = min(len(src_lines), len(trg_lines))
        src_lines = src_lines[:min_len]
        trg_lines = trg_lines[:min_len]

    filtered_src: List[str] = []
    filtered_trg: List[str] = []
    for s, t in zip(src_lines, trg_lines):
        if not s or not t:
            continue
        if max_strlen is not None:
            if s.count(" ") >= max_strlen or t.count(" ") >= max_strlen:
                continue
        filtered_src.append(s)
        filtered_trg.append(t)
        if max_samples is not None and len(filtered_src) >= max_samples:
            break

    return filtered_src, filtered_trg


class ParallelBatch:
    """Batch object compatible with both PyTorch training and legacy `batch.src` / `batch.trg`."""

    def __init__(self, src: torch.Tensor, trg: torch.Tensor, src_lengths: torch.Tensor):
        # Store as (seq_len, batch_size) on `.src` and `.trg` so `batch.src.transpose(0, 1)`
        # in demo_transformer.ipynb works identically, AND provide batch-first `.src_batch` / `.trg_batch`.
        self.src_batch = src
        self.trg_batch = trg
        self.src_lengths = src_lengths
        self.src = src.transpose(0, 1).contiguous()
        self.trg = trg.transpose(0, 1).contiguous()

    def __len__(self) -> int:
        return self.src_batch.size(0)


class ParallelTranslationDataset(Dataset):
    """Pre-tokenized and numericalized dataset for parallel sentences."""

    def __init__(
        self,
        src_sentences: Sequence[str | Sequence[str]],
        trg_sentences: Sequence[str | Sequence[str]],
        src_field: Field,
        trg_field: Field,
        max_len: int = 60,
    ):
        self.samples: List[Tuple[List[int], List[int]]] = []
        for s_raw, t_raw in zip(src_sentences, trg_sentences):
            s_toks = (src_field.preprocess(s_raw) if isinstance(s_raw, str) else list(s_raw))[:max_len]
            t_toks = (trg_field.preprocess(t_raw) if isinstance(t_raw, str) else list(t_raw))[: max_len - 2]
            if not s_toks or not t_toks:
                continue
            s_ids = src_field.vocab.encode(s_toks, add_sos=False, add_eos=False)
            t_ids = trg_field.vocab.encode(t_toks, add_sos=True, add_eos=True)
            self.samples.append((s_ids, t_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]


class BucketBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Pool-based length-sorted batch sampler (equivalent to MyIterator in demo_transformer.ipynb)."""

    def __init__(
        self,
        dataset: ParallelTranslationDataset,
        batch_size: int = 128,
        shuffle: bool = True,
        pool_multiplier: int = 100,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool_size = batch_size * pool_multiplier

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(indices)

        batches: List[List[int]] = []
        for start in range(0, len(indices), self.pool_size):
            pool = indices[start : start + self.pool_size]
            pool.sort(key=lambda idx: (len(self.dataset.samples[idx][0]), len(self.dataset.samples[idx][1])))
            pool_batches = [
                pool[i : i + self.batch_size]
                for i in range(0, len(pool), self.batch_size)
            ]
            if self.shuffle:
                rng.shuffle(pool_batches)
            batches.extend(pool_batches)

        for b in batches:
            yield b

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_parallel_batch(batch: List[Tuple[List[int], List[int]]]) -> ParallelBatch:
    """Pad a batch of (src_ids, trg_ids) pairs to the maximum length in the batch."""
    max_src = max(len(s) for s, _ in batch)
    max_trg = max(len(t) for _, t in batch)
    bs = len(batch)

    src_tensor = torch.full((bs, max_src), PAD_IDX, dtype=torch.long)
    trg_tensor = torch.full((bs, max_trg), PAD_IDX, dtype=torch.long)
    src_lengths = torch.zeros(bs, dtype=torch.long)

    for i, (s, t) in enumerate(batch):
        src_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        trg_tensor[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        src_lengths[i] = len(s)

    return ParallelBatch(src_tensor, trg_tensor, src_lengths)


def create_dataloader(
    dataset: ParallelTranslationDataset,
    batch_size: int = 128,
    shuffle: bool = True,
    bucket_sort: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader with length-sorted pooling for minimal padding."""
    if bucket_sort:
        sampler = BucketBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_parallel_batch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_parallel_batch,
        drop_last=False,
    )


def build_numpy_arrays(
    dataset: ParallelTranslationDataset,
    max_src_len: int = 60,
    max_trg_len: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ParallelTranslationDataset into padded NumPy arrays for TensorFlow/Keras training."""
    n = len(dataset)
    encoder_input = np.full((n, max_src_len), PAD_IDX, dtype=np.int32)
    decoder_input = np.full((n, max_trg_len), PAD_IDX, dtype=np.int32)
    decoder_target = np.full((n, max_trg_len), PAD_IDX, dtype=np.int32)

    for i, (s_ids, t_ids) in enumerate(dataset.samples):
        s_trunc = s_ids[:max_src_len]
        t_trunc = t_ids[: max_trg_len + 1]
        if len(t_trunc) < 2:
            continue
        t_in = t_trunc[:-1][:max_trg_len]
        t_out = t_trunc[1:][:max_trg_len]

        encoder_input[i, : len(s_trunc)] = s_trunc
        decoder_input[i, : len(t_in)] = t_in
        decoder_target[i, : len(t_out)] = t_out

    return encoder_input, decoder_input, decoder_target


def save_vocabs(src_field: Field, trg_field: Field, path: str) -> None:
    """Save source and target vocabularies to a JSON file."""
    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    payload = {
        "src": src_field.vocab.to_dict(),
        "trg": trg_field.vocab.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_vocabs(path: str) -> Tuple[Field, Field]:
    """Load source and target Fields from a saved JSON vocabulary file."""
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    src_field = Field(lang="en")
    trg_field = Field(lang="vi", init_token=SOS_TOKEN, eos_token=EOS_TOKEN)
    src_field.vocab = Vocab.from_dict(payload["src"])
    trg_field.vocab = Vocab.from_dict(payload["trg"])
    return src_field, trg_field
