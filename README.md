# Shallow Fusion Eval/POC

An integration and evaluation repository for testing shallow fusion techniques on medical domain ASR (Automatic Speech Recognition). Read the full motivation and technical details [**✨👉 HERE 👈✨**](https://donkeyanaphora.github.io/articles/article1/index.html)

## Overview

This proof-of-concept explores shallow fusion methods to improve medical transcription accuracy by combining acoustic and language models. The project aims to reduce Word Error Rate (WER) for medical terminology while maintaining overall transcription quality. For details on the model pretraining please refer to the model cards or the repo for the finetuning pipeline

## Related Resources
**Fine tuning pipeline and model cards**:

[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?logo=github)](https://github.com/donkeyanaphora/SHALLOW_FUSION)

[![Hugging Face Models](https://img.shields.io/badge/HuggingFace-Models-orange?logo=huggingface)](https://huggingface.co/cwestnedge/models) 


## Current Status

### Active Development
1. **Fusion Codebase Enhancement** - Optimizing for faster inference and seamless HuggingFace integration
2. **Metrics Development** - Creating granular error analysis tools to decompose error types
3. **Evaluation Pipeline** - Testing on larger medical ASR datasets beyond synthetic POC data

## Project Structure

```
shallow-fusion-eval/
├── data/
│   ├── input/      # POC text data
│   └── output/     # Generated .wav files
├── scripts/        # TTS conversion utilities
├── notebooks/      # Fusion implementation & evaluation
└── README.md
```

## Setup

*Documentation in progress*

## Data Sources

### Current Implementation
- **Synthetic dataset**: Text-to-speech generated audio-transcript pairs for initial testing

### Planned Data Sources
Real medical transcription datasets under consideration:

- [Shaip - Physician Dictation Data: Radiology](https://marketplace.databricks.com/details/8eb39dd5-ffc4-4e8d-8f89-25d91bf1774b/Shaip_Physician-Dictation-Data-Radiology)
- [John Snow Labs - Medical Transcription](https://marketplace.databricks.com/details/cd0b8356-8ae8-4178-a55b-7f69f040c0b8/John-Snow-Labs_Medical-Transcription)
- [Galileo - Medical Transcript 40](https://huggingface.co/datasets/galileo-ai/medical_transcription_40/viewer/default/train?row=0&views%5B%5D=train)

## Roadmap

### Core Challenge
Balancing improved WER for medical terminology against potential performance degradation on conversational elements when increasing the fusion parameter α.

### Technical Improvements

**Completed:**
- [x] Beam search decoding ([reference](https://huggingface.co/blog/mlabonne/decoding-strategies))
- [x] Prompt engineering for domain adaptation with GPT-2

**In Progress:**
- [ ] Cache `past_key_values` to avoid redundant encoder passes
- [ ] Develop adaptive gating mechanism for context-aware decoder influence
- [ ] Further fine-tune on medical domain (considering trade-offs with term forgetting)
- [ ] Fine-tune on conversational text (~10,000 samples)
- [ ] Fine-tune on spoken text (~10,000 samples)

### Evaluation Tasks
- [ ] Define metrics for conversational fillers
- [ ] Handle punctuation mismatches in evaluation sets
- [ ] Replace deprecated original dataset
- [ ] Quantify trade-offs between medical term accuracy and overall fluency

## Notes
This is an active research project. The main concern is improving WER for medical terms while managing performance on conversational filler terms that may be absent in the external LMs' pre-training data.