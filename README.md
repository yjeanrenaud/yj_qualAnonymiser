# yj_qualAnonymiser
A partical attempt to use NLP to classify word embeddings in texts, primarily narrative interview from qualitative social sciences, to make profound suggestions for anonymisation / pseudonymisation.
## Why?
Pseudonymisation or anonymsisation of Personal Identifiable Information (PII) is more than a pure redaction. At least for social scientists like I am one, it is important to preserve some information or categorial epistemological context while protecting informants of empirical qualitative social sciences.
## How?
My goal is to provide a tool that suggests parts of texts, usually qualitative interview transcripts, that should draw your attention. 
### NER and Word Emedding Similarity
Currently, my focus is on avoiding per-device configuration at all. Hence, I use multilingualy  NER (pre-trained transformers for Name Entity Recognition) and word embedding similarity (sentence-transformers) to get behind the words' embeddings.
### Regex and covabulary lists
Furthermore, as net step, I alsouse traditional regex (for email/phone/etc.) and include optional custom vocabulary lists (CSV files in a subfolder).
### Multilingual
I wanted to include as many languages as possible. The initial version of the Python script I could test English, German, and French transcripts of on various details. It looks quite promising.
### (Optional) GPU acceleration:
yj_piiMarker automatically uses CUDA GPU if available. It also uses Apple MPS if available (please, *help me testing* this! I currently lack the hardware)
If neither CUDA nor Apple MPS are available, it automatically falls back to CPU. The later is obviously slow but still usable.
## What for?
I want to include this in [noScribe](https://github.com/kaixxx/noScribe), my work horse for automated offline interview transcription on my daytime job as a social scientist.
If you're interested in my progress in implementing this noScribe, [check out my fork](https://github.com/yjeanrenaud/noScribe/).
