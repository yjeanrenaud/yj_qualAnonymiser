#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from __future__ import annotations
"""
Marks likely PII (Personal Identifiable Information) in free-form text (e.g., interview transcripts) by wrapping spans like:

   "I live in Munich..."
-->"I live in *** Munich [city] *** ..."

# DISCLAIMER # 
No automated PII detection is perfect. Always review redactions before sharing externally. No liability whatsover.
The output from this script are intented as a starting point for anonymising / pseudonymising your qualitative data. 
The script works multilingualy (currently tested with English, German, and French. It uses multilingual NER (pre-trained transformers for Name Entity Recognition) word embedding similarity (sentence-transformers) to get behind the words' embeddings.
Furthermore, it also employs traditional regex for (email/phone/etc.) and optional custom vocabulary lists (CSV files in a subfolder, see on VOCAB below.).

(optional) GPU acceleration:
yj_piiMarker automatically uses CUDA GPU if available. It uses Apple MPS if available.
If neither CUDA nor Apple MPS are available, it automatically falls back to CPU (slow).
No device configuration whatsoever is required.

# INSTALL #
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -U pip

# Core deps #
pip install -U transformers torch sentence-transformers pandas numpy

# Optional: faster CPU inference (may or may not be available on your platform)
# pip install -U accelerate

# NVIDIA GPU #
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# CPU / Apple Silicon #
pip install torchvision torchaudio

# Intel NPU
pip install -U "optimum-intel[openvino]" openvino

# Custom Vocab CSV format #
Create a folder like:
  vocab/
    places.csv
    people.csv
    any-name.csv

Each CSV must be:
  phrase,tag

Example vocab/places.csv:
phrase,tag
New York,city
Berlin,city
Alexanderplatz,location

Example vocab/people.csv:
phrase,tag
John Doe,name
Dr. Müller,name

Notes:
- "phrase" is matched by embedding similarity (robust to minor variations).
- You can include multiple languages in the same vocab.

# Usage #
# Direct text from cli
python pii_marker_multilingual.py --text "I live in New York and my email is test@example.com"

# File operations
python pii_marker_multilingual.py --in transcript.txt --out marked.txt

# Stdi/o
cat transcript.txt | python pii_marker_multilingual.py > marked.txt

# With custom vocab folder
python pii_marker_multilingual.py --in transcript.txt --vocab_dir vocab --out marked.txt

# Tuning #
Key thresholds:
  --vocab_threshold   default 0.78  (higher is stricter)
  --proto_threshold   default 0.70  (higher is stricter)

If you get too many false positives:
  increase thresholds (e.g., 0.82 / 0.75)

If you miss things:
  decrease thresholds (e.g., 0.72 / 0.65)

# DISCLAIMER # 
No automated PII detection is perfect. Always review redactions before sharing externally.
"""

import argparse
import glob
import os
import re
import sys
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Automatic device detection
# -------------------------------------------------
def detect_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return {"backend": "torch", "hf_device": 0, "st_device": "cuda"}

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS acceleration")
        return {"backend": "torch", "hf_device": 0, "st_device": "mps"}
    try:
        import openvino as ov
        devs = ov.Core().available_devices
        if any(d.upper().startswith("NPU") for d in devs) or "NPU" in [d.upper() for d in devs]:
            print("Using Intel NPU via OpenVINO")
            return {"backend": "openvino", "ov_device": "NPU"}
    except Exception:
        pass
   
    print("Using CPU")
    return {"backend": "torch", "hf_device": -1, "st_device": "cpu"}


# -------------------------------------------------
# Regex detectors
# -------------------------------------------------
REGEX_PATTERNS = [
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), 5),
    ("url", re.compile(r"\bhttps?://[^\s]+\b", re.I), 5),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 5),
    ("phone", re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)"), 4),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b"), 4),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), 5),
]
TOKEN_RE = re.compile(r"\b[\w'’-]+\b", re.UNICODE)

# -------------------------------------------------
# NER label mapping
# -------------------------------------------------
NER_LABEL_MAP = {
    "PER": "name",
    "PERSON": "name",
    "LOC": "location",
    "GPE": "city",
    "ORG": "organization",
}


# -------------------------------------------------
# Span structure
# -------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    tag: str
    priority: int

    def length(self):
        return self.end - self.start


# -------------------------------------------------
# Model loading
# -------------------------------------------------
def load_models(device_hf, device_st):
    model_id = "Davlan/xlm-roberta-base-ner-hrl"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    ner_pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device_hf,
    )

    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=device_st,
    )
    return ner_pipe, embedder
   
def load_models_openvino(ov_device: str):
    from transformers import AutoTokenizer, pipeline
    from optimum.intel.openvino import OVModelForTokenClassification

    model_id = "Davlan/xlm-roberta-base-ner-hrl"

    model = OVModelForTokenClassification.from_pretrained(model_id, device=ov_device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    ner_pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
    )
    return ner_pipe, embedder

def load_vocab(vocab_dir: Optional[str]) -> List[Tuple[str, str]]:
    entries = []
    if not vocab_dir:
        return entries
    if not os.path.isdir(vocab_dir):
        return entries

    for path in glob.glob(os.path.join(vocab_dir, "*.csv")):
        df = pd.read_csv(path)
        if "phrase" not in df.columns or "tag" not in df.columns:
            continue

        for _, row in df.iterrows():
            phrase = str(row["phrase"]).strip()
            tag = str(row["tag"]).strip()
            if phrase and tag and phrase.lower() != "nan" and tag.lower() != "nan":
                entries.append((phrase, tag))

    return entries

def build_vocab_index(embedder, vocab_entries: List[Tuple[str, str]]):
    if not vocab_entries:
        return {"buckets": {}, "max_words": 0}

    buckets = {}
    max_words = 0

    for phrase, tag in vocab_entries:
        n_words = len(TOKEN_RE.findall(phrase))
        if n_words == 0:
            continue
        max_words = max(max_words, n_words)
        buckets.setdefault(n_words, []).append((phrase, tag))

    out = {}
    for n_words, items in buckets.items():
        phrases = [p for p, _ in items]
        tags = [t for _, t in items]

        embeddings = embedder.encode(
            phrases,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        out[n_words] = {
            "phrases": phrases,
            "tags": tags,
            "embeddings": embeddings,
        }

    return {"buckets": out, "max_words": max_words}

def vocab_spans(text, embedder, vocab_index, threshold=0.78, debug=False, top_k=3):
    spans = []

    if not embedder or not vocab_index or not vocab_index["buckets"]:
        return spans

    tokens = list(TOKEN_RE.finditer(text))
    if not tokens:
       if debug:
           print("\n+++ Vocab debug +++")
           print("No tokens found")
           print("+++ /Vocab debug +++\n")
       return spans

    for n_words, bucket in vocab_index["buckets"].items():
        if len(tokens) < n_words:
            continue

        candidate_texts = []
        candidate_positions = []

        for i in range(len(tokens) - n_words + 1):
            start = tokens[i].start()
            end = tokens[i + n_words - 1].end()
            candidate = text[start:end].strip()

            if not candidate:
                continue

            candidate_texts.append(candidate)
            candidate_positions.append((start, end))

        if not candidate_texts:
            continue

        cand_emb = embedder.encode(
            candidate_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        sims = cand_emb @ bucket["embeddings"].T

        if debug:
            print(f"\n+++ Vocab bucket: n_words={n_words}, vocab_items={len(bucket['phrases'])}, candidates={len(candidate_texts)} +++")

        for row_idx, ((start, end), candidate) in enumerate(zip(candidate_positions, candidate_texts)):
            row = sims[row_idx]
            top_ids = np.argsort(-row)[:top_k]

            if debug:
                print(f"\nDEBUG candidate: '{candidate}' [{start}:{end}]")
                for rank, idx in enumerate(top_ids, start=1):
                    print(
                        f"  #{rank} score={row[idx]:.4f} "
                        f"phrase='{bucket['phrases'][idx]}' "
                        f"tag='{bucket['tags'][idx]}'"
                    )

            best_idx = np.argmax(sims, axis=1)
            best_scores = sims[np.arange(len(candidate_texts)), best_idx]

#        for (start, end), score, idx in zip(candidate_positions, best_scores, best_idx):
#            if score >= threshold:
#                spans.append(Span(start, end, bucket["tags"][idx], 3))
            if best_score >= threshold:
                spans.append(Span(start, end, bucket["tags"][best_idx], 3))
                if debug:
                    print("  -> ACCEPTED")
            elif debug:
                print("  -> rejected")

    if debug:
        print("\n+++ /Vocab debug +++\n")

    return spans

# -------------------------------------------------
# Regex detection
# -------------------------------------------------
def regex_spans(text, debug=False):
    spans = []

    if debug:
        print("\n+++ Regex raw +++")
        print(text)
        print("+++ /Regex raw +++\n")

    for tag, pat, pr in REGEX_PATTERNS:
        matches = list(pat.finditer(text))

        if debug:
            print(f"Regex pattern '{tag}': {len(matches)} match(es)")

        for m in matches:
            span = Span(m.start(), m.end(), tag, pr)
            spans.append(span)

            if debug:
                print(
                    f"  [{m.start()}:{m.end()}] "
                    f"'{text[m.start():m.end()]}' -> tag='{tag}', priority={pr}"
                )

    if debug:
        print("+++ /Regex matches +++\n")

    return spans


# -------------------------------------------------
# NER detection
# -------------------------------------------------
def ner_spans(pipe, text, debug=False):
    spans = []
    ents = pipe(text)

    if debug:
        print("\n+++ NER raw +++")
        for ent in ents:
            print(ent)
        print("+++ /NER raw +++\n")

    for ent in ents:
        label = ent.get("entity_group", "")
        tag = NER_LABEL_MAP.get(label)
        start = ent.get("start")
        end = ent.get("end")

        if tag and start is not None and end is not None and end > start:
            spans.append(Span(start, end, tag, 2))

    return spans


# -------------------------------------------------
# Resolve overlaps
# -------------------------------------------------
def resolve(spans):
    spans = [
        s for s in spans
        if s.start is not None and s.end is not None and s.end > s.start
    ]

    spans = sorted(spans, key=lambda s: (s.start, -s.priority, -s.length()))
    out = []

    for s in spans:
        if not out or s.start >= out[-1].end:
            out.append(s)
        else:
            prev = out[-1]
            if s.priority > prev.priority:
                out[-1] = s

    return out


# -------------------------------------------------
# Wrap output
# -------------------------------------------------
def wrap(text, spans):
    cursor = 0
    out = []
    for s in spans:
        out.append(text[cursor:s.start])
        out.append(f"*** {text[s.start:s.end]} [{s.tag}] ***")
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)
# ------------------------------------------------- 
# debug helper
# ------------------------------------------------- 
def debug_pair_scores(embedder, queries, vocab_phrases):
    q = embedder.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    v = embedder.encode(
        vocab_phrases,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sims = q @ v.T

    for i, query in enumerate(queries):
        print(f"\nQUERY: {query}")
        order = np.argsort(-sims[i])
        for j in order[:5]:
            print(f"  score={sims[i, j]:.4f}  vocab='{vocab_phrases[j]}'")

# -------------------------------------------------
# Main PII detection
# -------------------------------------------------
def mark_pii(text, ner_pipe, embedder=None, vocab_index=None, vocab_threshold=0.78, debug=False):
    spans = []
    spans.extend(regex_spans(text, debug=debug))                         # step 1: regex
    spans.extend(ner_spans(ner_pipe, text, debug=debug))                # step 2: ner
    spans.extend(vocab_spans(text, embedder, vocab_index, vocab_threshold, debug=debug))  # step 3: sentence-transformer vocab similarity 
    spans = resolve(spans)
    return wrap(text, spans)


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--text")
    g.add_argument("--in", dest="infile")
    ap.add_argument("--out", dest="outfile")
    ap.add_argument("--vocab_dir")
    ap.add_argument("--vocab_threshold", type=float, default=0.78)
    ap.add_argument("--debug", action="store_true", help="Print NER and vocab similarity debug output")

    args = ap.parse_args()

    if args.text:
        text = args.text
    elif args.infile:
        with open(args.infile, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    accel = detect_device()

    if accel["backend"] == "openvino":
        ner_pipe, embedder = load_models_openvino(accel["ov_device"])
    else:
        ner_pipe, embedder = load_models(accel["hf_device"], accel["st_device"])

    #result = mark_pii(text, ner_pipe)
    vocab_entries = load_vocab(args.vocab_dir)
    vocab_index = build_vocab_index(embedder, vocab_entries) if vocab_entries else None

    result = mark_pii(
        text,
        ner_pipe,
        embedder=embedder,
        vocab_index=vocab_index,
        vocab_threshold=args.vocab_threshold,
        debug=args.debug,
    )

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)

if __name__ == "__main__":
    main()
