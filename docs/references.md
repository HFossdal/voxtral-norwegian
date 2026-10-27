# References

Prior work this project uses or follows, with what we took from each and
where it is used. Citation details were checked on 2026-10-20 (arXiv API,
ACL Anthology, Hugging Face); entries added later give their own check
date.

## Models

**[Voxtral]** A. H. Liu, A. Ehrenberg, A. Lo, C. Denoix, C. Barreau,
G. Lample, et al. *Voxtral.* arXiv:2507.13264, 2025.
- Model: `mistralai/Voxtral-Mini-3B-2507` (Apache-2.0). Baseline and
  fallback model. Its transcription prompt and `lang:xx` hint.
- Fine-tuned offline on the same data as the Realtime model, with its own
  transcription prompt as the training input; the final model
  (5.91% test WER).

**[Voxtral Realtime]** Mistral AI: A. H. Liu, A. Ehrenberg, A. Lo, C.-Y. Sun,
et al. *Voxtral Realtime.* arXiv:2602.11298, 2026.
- Model: `mistralai/Voxtral-Mini-4B-Realtime-2602` (Apache-2.0). Target
  model.
- Taken: the frame-synchronous target scheme (one token per 80 ms frame,
  pad token, word token emitted once a word is fully heard and the delay has
  passed, no extra word token for words sharing a frame), which our
  training sequences reproduce (`voxnor.realtime_stream`).

**[Voxtral Realtime Arabic]** Mistral AI. `mistralai/Voxtral-Mini-4B-Realtime-Arabic`,
Hugging Face model, created 2026-10-08 (checked 2026-10-24).
- A language-adapted version of the Realtime model published by Mistral:
  part of why adapting the open model to a new language is a relevant
  result. Not used.

**[Whisper]** A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey,
I. Sutskever. *Robust Speech Recognition via Large-Scale Weak Supervision.*
arXiv:2212.04356, 2022; ICML 2023 (arXiv API checked 2026-10-25).
- Voxtral Mini's audio encoder follows Whisper's (30 s input windows,
  `out_proj`/`fc1`/`fc2` layer names), which is why the encoder LoRA
  pattern also matches those names and every Mini clip is padded to
  30 s. NB-Whisper (below) is a fine-tuned Whisper.

**[Voxtral 3]** Mistral AI. *Voxtral 3.* News post, mistral.ai/news/voxtral-3,
2026-10-20 (read the same day). API models (Small / Mini Transcribe 3,
Realtime Transcribe 3, Speech 3); no open weights named.

**[Mistral transcription API]** Mistral AI documentation, "Transcribe files"
(docs.mistral.ai/studio/audio/speech-to-text/transcribe-files) and "API
pricing" (docs.mistral.ai/inference/pricing), read 2026-10-23. Model
`voxtral-mini-transcribe-3`, `POST /v1/audio/transcriptions` with `model`,
`file` and an optional ISO 639-1 `language`; $0.0035 per audio minute; no
temperature setting documented. Used by `voxnor.mistral_api`.
- Context only: newer Mistral speech models whose benchmarks list
  Norwegian. Not used in training; `voxtral-mini-transcribe-3` is scored
  as a reference on validation and test.

**[DSM]** N. Zeghidour, E. Kharitonov, M. Orsini, V. Volhejn,
G. de Marmiesse, E. Grave, et al. *Streaming Sequence-to-Sequence Learning
with Delayed Streams Modeling.* arXiv:2509.08753, 2025.
- The delayed-streams approach Voxtral Realtime builds on. Background only.

**[NB-wav2vec2]** National Library of Norway (NbAiLab).
`NbAiLab/nb-wav2vec2-300m-bokmaal-v2`, Hugging Face model, Apache-2.0.
- Used for CTC forced alignment of NPSC transcripts to get word end times
  (`voxnor.align`). Not used in evaluation.

**[wav2vec2-base-960h]** Meta AI. `facebook/wav2vec2-base-960h`,
Hugging Face model.
- English forced alignment in `scripts/probe_realtime.py`, only to check our
  word-placement formula against the Realtime model's own output.

**[NB-Whisper]** National Library of Norway (NbAiLab). `NbAiLab/nb-whisper-*`,
Hugging Face models, Apache-2.0.
- Reference point only, not a fair comparison. Scored on
  validation and test by `scripts/run_reference.py`: `nb-whisper-large` and
  `nb-whisper-large-verbatim`. The model card (checked 2026-10-22) lists
  NPSC, Stortinget speeches, NST, NRK subtitles and audiobooks among the
  training data, about 66,000 hours in total; its `test_results.json`
  reports 9.46% WER (12.74% unnormalised) on a Stortinget test set, with
  its own normalisation.

## Fine-tuning Voxtral Realtime

**[Zheng 2026]** X. Zheng. *Look Less, Hear Better: Jointly Rewarded GRPO
for Streaming ASR.* arXiv:2609.18333, 2026.
- The closest prior work to our Realtime fine-tuning: supervised LoRA
  fine-tuning of Voxtral-Mini-4B-Realtime-2602 (English, about 175k hours), before a
  reinforcement-learning stage we do not use.
- Taken: word tokens placed from frame `ceil(end / 80 ms) + delay`
  (their Eq. 1) using forced-aligned end times; LoRA on the language
  decoder only, with the audio encoder frozen (they used r = α = 256);
  padding frames after the transcript before the end.
- Differences: we adapt to a language the model does not support, with
  32.6 h of NPSC (101.6 h with SSC), and train at a single delay (480 ms)
  instead of sampling 80–2400 ms.

## Data

**[NPSC]** P. E. Solberg and P. Ortiz. *The Norwegian Parliamentary Speech
Corpus.* Proceedings of LREC 2022.
- The corpus (CC0).

**[NPSC Bokmål subset]** P. E. Solberg, P. Ortiz, P. Parsons, T. Svendsen,
G. Salvi. *Improving Generalization of Norwegian ASR with Limited
Linguistic Resources.* NoDaLiDa 2023, pp. 508–517.
- `scribe-project/npsc_nb`: the Bokmål, under-15-second version of NPSC we
  train and evaluate on, including `standardized_text`.

**[SSC]** P. E. Solberg, P. Beauguitte, P. E. Kummervold, F. Wetjen. *A
Large Norwegian Dataset for Weak Supervision ASR.* RESOURCEFUL-2023,
pp. 48–52. Stortinget Speech Corpus v1.0 (CC0), dataset card `SSC_1.pdf`
from Språkbanken, checked 2026-10-23. Downloaded from the WebDataset mirror
`Aalto-Speech-Synthesis/stortinget_speech_corpus_v1.0` on Hugging Face
(CC0, at a pinned revision; checked 2026-10-25).
- Used as extra training data: clips where the proceedings text and SSC's
  ASR agree, with NPSC validation/test speakers and meeting dates left out
  (69.4 h, `voxnor.ssc`, `scripts/build_ssc.py`).

**[ParlaMint]** T. Erjavec, M. Ogrodniczuk, P. Osenova, N. Ljubešić,
K. Simov, A. Pančur, et al. *The ParlaMint corpora of parliamentary
proceedings.* Language Resources and Evaluation 57, pp. 415–448, 2023,
doi:10.1007/s10579-021-09574-0. T. Erjavec, M. Kopp, N. Ljubešić,
T. Kuzman, P. Rayson, et al. *ParlaMint II: advancing comparable
parliamentary corpora across Europe.* Language Resources and Evaluation
59(3), pp. 2071–2102, 2025, doi:10.1007/s10579-024-09798-w (both checked on
Crossref 2026-10-26). Data: `Samples/ParlaMint-NO/ParlaMint-NO-listPerson.xml`
from github.com/clarin-eric/ParlaMint at a pinned commit.
- Its Norwegian person list gives the speaker IDs that SSC uses. NPSC's
  validation and test speakers are matched to it by name, and those 118
  people are left out of the SSC training data
  (`scripts/make_ssc_lists.py`, `voxnor.ssc`). No proceedings text is used.

**[NST]** Nordisk Språkteknologi, Norwegian acoustic database (read speech,
recorded 1999), preserved by Språkbanken. `NbAiLab/NST` on Hugging Face
(Apache-2.0), checked 2026-10-23.
- Considered as extra training data; not used.

## Methods and tools

**Data-parallel training.** S. Li et al. *PyTorch Distributed: Experiences
on Accelerating Data Parallel Training.* PVLDB 13(12), 2020. Describes
PyTorch's `DistributedDataParallel`: one model copy per GPU and an
all-reduce of the gradients each step, overlapped with the backward pass in
buckets. We use the same scheme with one all-reduce after the backward pass
instead of PyTorch's wrapper, and torchrun (`torch.distributed.run`) to
start the processes (`voxnor.distributed`). Built and tested on CPU
only; every reported run trained on a single GPU.

**CTC forced alignment.** Viterbi alignment over the CTC topology (blanks
optional except between repeated labels), as in A. Graves et al.,
*Connectionist Temporal Classification*, ICML 2006, and as implemented in
`torchaudio.functional.forced_align`. Our own implementation:
`voxnor.align.ctc_viterbi`.

**LoRA.** E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language
Models.* ICLR 2022. Implemented with Hugging Face `peft`.

**AdamW.** I. Loshchilov and F. Hutter. *Decoupled Weight Decay
Regularization.* ICLR 2019. `torch.optim.AdamW`, used for fine-tuning.

**Bootstrap intervals.** Corpus-level WER with bootstrap resampling of
utterances, and the paired bootstrap for comparing systems, following
M. Bisani and H. Ney, *Bootstrap estimates for confidence intervals in ASR
performance evaluation*, ICASSP 2004.

**Speed perturbation.** T. Ko, V. Peddinti, D. Povey and S. Khudanpur.
*Audio Augmentation for Speech Recognition.* Interspeech 2015, pp. 3586–3589,
doi:10.21437/Interspeech.2015-711 (ISCA Archive entry checked 2026-10-21).
They resample the raw audio at speed factors 0.9, 1.0 and 1.1, which changes
tempo and pitch together, and report a 4.3% average relative WER improvement
on four tasks. We use the same factors, but draw one factor per clip and
step instead of storing three copies of the data (`voxnor.augment`).

**Noise augmentation.** Adding noise to clean training audio so the model
sees more varied input, as in A. Hannun et al., *Deep Speech: Scaling up
end-to-end speech recognition*, arXiv:1412.5567, 2014 (they superimpose
recorded noise; arXiv API checked 2026-10-25). We add white Gaussian noise
at 10–30 dB SNR with probability 0.5: the parliament audio is clean,
and recorded noise (e.g. MUSAN) would need another dataset.

## Software

Exact versions are pinned in `pyproject.toml` and recorded in every run's
`environment.json`.
- PyTorch (`torch`): training, mixed precision, `torch.distributed`.
- Hugging Face `transformers` (Voxtral model classes and processors),
  `peft` (LoRA), `datasets` and `huggingface_hub` (data and model download).
- `mistral-common`: Voxtral's tokenizer and transcription requests.
- `jiwer`: word and character alignments for WER and CER
  (`voxnor.metrics`, `voxnor.error_analysis`).
- `soundfile` (audio decoding), `scipy` (`resample_poly` for speed
  perturbation), `pandas`, `matplotlib` (figures), `requests` (Mistral API).
