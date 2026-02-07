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
from transformers import pipeline
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
    ner_pipe = pipeline(
        "token-classification",
        model="Davlan/xlm-roberta-base-ner-hrl",
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

    # This compiles the model for the given OpenVINO device (CPU/GPU/NPU).
    model = OVModelForTokenClassification.from_pretrained(model_id, device=ov_device)
    tok = AutoTokenizer.from_pretrained(model_id)

    ner_pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tok,
        aggregation_strategy="simple",
    )
   return ner_pipe, embedder


# -------------------------------------------------
# Regex detection
# -------------------------------------------------
def regex_spans(text):
    spans = []
    for tag, pat, pr in REGEX_PATTERNS:
        for m in pat.finditer(text):
            spans.append(Span(m.start(), m.end(), tag, pr))
    return spans


# -------------------------------------------------
# NER detection
# -------------------------------------------------
def ner_spans(pipe, text):
    spans = []
    for ent in pipe(text):
        label = ent.get("entity_group", "")
        tag = NER_LABEL_MAP.get(label)
        if tag:
            spans.append(
                Span(ent["start"], ent["end"], tag, 2)
            )
    return spans


# -------------------------------------------------
# Resolve overlaps
# -------------------------------------------------
def resolve(spans):
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
# Main PII detection
# -------------------------------------------------
def mark_pii(text, ner_pipe):
    spans = []
    spans.extend(regex_spans(text))
    spans.extend(ner_spans(ner_pipe, text))
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

    args = ap.parse_args()

    if args.text:
        text = args.text
    elif args.infile:
        with open(args.infile, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    device_hf, device_st = detect_device()
    if accel["backend"] == "openvino":
        ner_pipe, _ = load_models_openvino(accel["ov_device"])
       #ner_pipe, _ = load_models(device_hf, device_st)
    else:
        ner_pipe, _ = load_models_torch(accel["hf_device"], accel["st_device"])    

    result = mark_pii(text, ner_pipe)

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
