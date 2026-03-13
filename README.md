# yj_qualAnonymiser

A partial attempt to use NLP to classify word embeddings in texts, primarily in narrative interviews from qualitative social sciences, to make meaningful suggestions for anonymisation / pseudonymisation. It runs completely locally and offline.
--
## Table of Contents

- [Why?](#why)
- [How?](#how)
  - [NER and Word Embedding Similarity](#ner-and-word-embedding-similarity)
  - [Regex and Vocabulary Lists](#regex-and-vocabulary-lists)
  - [Multilingual](#multilingual)
  - [Optional GPU and NPU Acceleration](#optional-gpu-and-npu-acceleration)
- [What for?](#what-for)
- [Installation](#installation)
  - [Custom Vocab CSV format](#custom-vocab-csv-format)
- [Usage](#usage)
- [Todo](#todo)
- [License and Citation](#license-and-citation)
--
## Why?

Pseudonymisation or anonymisation of *Personally Identifiable Information* (PII) is more than pure redaction. For social scientists like me, it is important to preserve certain information or categorical epistemological context while still protecting informants in empirical qualitative social science research.

## How?

My goal is to provide a tool that suggests parts of texts, usually qualitative interview transcripts, that should draw your attention.

### NER and Word Embedding Similarity

Currently, my focus is on avoiding per-device configuration altogether. Therefore, I use multilingual NER (pre-trained transformers for Named Entity Recognition) and word embedding similarity (sentence-transformers) to capture semantic relationships behind words.

### Regex and Vocabulary Lists

As a next step, I also use traditional regex patterns (for email/phone/etc.) and include optional custom vocabulary lists (CSV files in a subfolder).

### Multilingual

I aim to support as many languages as possible. With the initial version of the Python script, I tested English, German, and French transcripts across various cases, and results look promising. (*help me with testing further languages, too*)

### (Optional) GPU and NPU Acceleration

`yj_piiMarker` automatically uses a CUDA GPU if available. It also supports Apple MPS if available (*help with testing is appreciated*. I currently lack compatible hardware).

If neither CUDA nor Apple MPS are available, it automatically falls back to CPU. The latter is slower but still usable.

I added support for the new Intel CPUs with dedicated NPU. This might bring some performance boosts, too.

## What for?

I plan to include this in [noScribe](https://github.com/kaixxx/noScribe), my workhorse for automated offline interview transcription in my daytime job as a social scientist.
If you're interested in my progress integrating this into noScribe, [check out my fork of Kai's noScribe](https://github.com/yjeanrenaud/noScribe/).

## Installation
1. set up python
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   ```
2. Install core deps
   
   `pip install -U transformers torch sentence-transformers pandas numpy`   
4. *Optional*: faster CPU inference (may or may not be available on your platform)
   
   `pip install -U accelerate`
6. *Optional*: for NVIDIA GPU
   
   `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
8. *Optional*: CPU / Apple Silicon
   
   `pip install torchvision torchaudio`
10. *Optional but recommended*: Intel NPU
    
    These co-processors (technically wrong, I know) are included in newer CPUs from Intel. I think it was since the first Intel Core Ultra Series (Meteor Lake), hence 2023. So if your Computer is Intel Based and newer, this is a good option (even though proprietary blobs are required)
   `pip install -U "optimum-intel[openvino]" openvino`

### Custom Vocab CSV format
also optional, but helpful: Create a folder like this:
  ```
  vocab/
    places.csv
    people.csv
    any-name.csv
  ```

- Each CSV must be formated like this:
  `phrase,tag`

- Example vocab/places.csv:
  ```
  phrase,tag
  New York,city
  Berlin,city
  Alexanderplatz,location
  ```
- Example vocab/people.csv:
  ```
  phrase,tag
  John Doe,name
  Dr. Müller,name
  ```

Notes on vocabolaries:
- "phrase" is matched by embedding similarity (robust to minor variations).
- You can include multiple languages in the same vocab file.
- Tis feature is ment for words and expressions that might be very, very rare, very field specific or something you expect to occure in your material.

## Usage
- basic test ifthe mechanism works (and downloading the models for the first run (around 3 GiB)
- ```python3 yj_piiMarker.py --text "Marcelle was my grandmother. She lived in Lausanne, a city in Switzerland, and also knew the Rüdisühlis. where I would visit her in from time to time as a child. Although she passed away about 35 years ago, I still remember her and her husband quite vividly." --vocab_dir vocab --debug```
- In and out files

  `--in transcript.txt --out marked.txt`
- With custom vocab folder

  `--vocab_dir vocab`
- Debugging output

  `--debug`
- Tuning
  Key thresholds:

  `--vocab_threshold   default 0.78  (higher is stricter)`
  `-proto_threshold   default 0.70  (higher is stricter)`

  If you get too many false positives:

  `increase thresholds (e.g., 0.82 / 0.75)`

  If you miss things:

  `decrease thresholds (e.g., 0.72 / 0.65)`


# Todo

- hardening
- portability

# License and Citation

This work is licensed under the AGPL-3. Hence, it's open source.

If you use this software in academic work, please cite:

Jeanrenaud, Yves (2026). *yj_qualAnonymiser*. https://github.com/yjeanrenaud/yj_qualAnonymiser/.


