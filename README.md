# yj_qualAnonymiser

A partial attempt to use NLP to classify word embeddings in texts—primarily narrative interviews from qualitative social sciences—to make meaningful suggestions for anonymisation / pseudonymisation.

## Why?

Pseudonymisation or anonymisation of *Personally Identifiable Information *(PII) is more than pure redaction. For social scientists like me, it is important to preserve certain information or categorical epistemological context while still protecting informants in empirical qualitative social science research.

## How?

My goal is to provide a tool that suggests parts of texts, usually qualitative interview transcripts, that should draw your attention.

### NER and Word Embedding Similarity

Currently, my focus is on avoiding per-device configuration altogether. Therefore, I use multilingual NER (pre-trained transformers for Named Entity Recognition) and word embedding similarity (sentence-transformers) to capture semantic relationships behind words.

### Regex and Vocabulary Lists

As a next step, I also use traditional regex patterns (for email/phone/etc.) and include optional custom vocabulary lists (CSV files in a subfolder).

### Multilingual

I aim to support as many languages as possible. With the initial version of the Python script, I tested English, German, and French transcripts across various cases, and results look promising. (*help me with testing further languages, too*)

### (Optional) GPU Acceleration

`yj_piiMarker` automatically uses a CUDA GPU if available. It also supports Apple MPS if available (*help with testing is appreciated*. I currently lack compatible hardware).

If neither CUDA nor Apple MPS are available, it automatically falls back to CPU. The latter is slower but still usable.

## What for?

I plan to include this in [noScribe](https://github.com/kaixxx/noScribe), my workhorse for automated offline interview transcription in my daytime job as a social scientist.
If you're interested in my progress integrating this into noScribe, [check out my fork of Kai's noScribe](https://github.com/yjeanrenaud/noScribe/).

## License and Citation

This work is licensed under the PolyForm Noncommercial License. In short, this means, you are free to use and modify this, as long as you mention me and this repository. Any commercial use is not permitted without a separate agreement.

If you use this software in (academic= work, please cite:
Jeanrenaud, Yves (2026). *yj_qualAnonymiser*. https://github.com/yjeanrenaud/yj_qualAnonymiser/.
