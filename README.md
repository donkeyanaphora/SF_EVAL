# SHALLOW FUSION EVAL – Next Steps

## Articles – Next Steps
- replace original dataset (no longer available) have a few in the general to do sections that could be useful maybe just start out with synthetic POC
- If we see an improvement on those despite a drop in **WER** in aggregate, then maybe we fine-tune on conversational text just to develop prose/address domain mismatch issues
- Fine-tune on spoken text (maybe **10 000** samples)… <https://huggingface.co/datasets/MLCommons/peoples_speech/viewer/clean/train?row=2>  
- New eval data? <https://www.shaip.com/offerings/physician-dictation-audio-data-medical-data-catalog/>

> I think it’s OK to say I generated a small synthetic dataset as a proof of concept, but ideally we would try this on larger-scale data—though it’s hard to come by.

---

## Problem
Concerns include improvements in **WER** for medical terms at the cost of performance dips as we increase *alpha* on conversational filler terms that are absent in the external LMs’ pre-training.

## Solutions
- [ ] Try beam search  
- [ ] Fine-tune the LLM on domain-appropriate language (could cause forgetting of medical terms)  
- [x] Append a prompt to GPT-2 with tokens to push it closer to domain (probably only works well on models with few-shot capabilities, but maybe we could get clever)  
- [ ] Learn dynamic lambda or a gated mechanism that controls influence of each decoder

---

## General To-Do
- [ ] Beam search: <https://huggingface.co/blog/mlabonne/decoding-strategies>  
- [ ] Cache `past_key_values` so we don’t have to pass through the encoder at each decoding step  
- [ ] Create data on radiology reports/memos: <https://marketplace.databricks.com/details/8eb39dd5-ffc4-4e8d-8f89-25d91bf1774b/Shaip_Physician-Dictation-Data-Radiology>  
- [ ] Convert this to audio with our pipeline and evaluate: <https://huggingface.co/datasets/galileo-ai/medical_transcription_40/viewer/default/train?row=0&views%5B%5D=train>  
- [ ] Add conversational fillers to eval set or use real data  