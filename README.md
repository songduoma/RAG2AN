# RAG²AN – Retrieval-Augmented Generator & Discriminator for Fake News GAN

> Research use only. This code is for fake-news detection and adversarial generation research. Do not use it to spread misinformation.

RAG²AN ties a generator with retrieval-augmented generation (RAG) and an encoder-based discriminator into a simple GAN-style loop. The generator crafts suspect news with external context; the discriminator judges plausibility and surfaces suspicious words for feedback, iteratively improving the discriminator.

## Project Structure
- `gan_training.py`: Main training loop for loading data, calling the generator, training the discriminator, and saving checkpoints.
- `train.sh`: Convenience launcher; tweak environment variables to change dataset, rounds, and models.
- `generator.py`: Default generator is `Qwen/Qwen2.5-7B-Instruct`, with built-in Wikipedia RAG and feedback prompt support.
- `discriminator.py`: Encoder-based discriminator (DeBERTa) that can return suspicious words via attention.
- `search.py`: Google Search RAG via SerpAPI.
- `inspect_gan_run.py`: Inspect output directories to browse round results and stats.

## Requirements & Install
- Python 3.10 with CUDA 12.1 GPU (Qwen2.5-7B-Instruct typically needs 16GB+ VRAM; use a smaller model like `gpt2` if constrained).
- HF token: Accept Qwen 7B terms on Hugging Face and set `HF_TOKEN`.
- Google RAG: set `SERPAPI_API_KEY` (package `google-search-results` is included).
- OpenAI / Gemini: only needed if you call the helpers in `utils.py` (`OPENAI_API_KEY` / `GEMINI_API_KEY`).

Install:
```bash
cd RAG2AN
# activate your venv/conda env first if needed
pip install -r requirements.txt   # includes PyTorch cu121 extra-index
```
> If you are not on CUDA 12.1, change the `--extra-index-url` at the top of `requirements.txt` or install torch/torchvision/torchaudio that match your CUDA.

## Quickstart
Default run (CNN/DailyMail subset, Wiki RAG, 10 GAN rounds):
```bash
cd RAG2AN
./train.sh
```

Customize via env vars (example: fewer rounds, custom output path, enable disc RAG):
```bash
DATASET_NAME=cnn_dailymail \
DATASET_CONFIG=3.0.0 \
DATASET_SPLIT=train[:64] \
NUM_ROUNDS=3 \
DISC_EPOCHS=1 \
BATCH_SIZE=4 \
RAG_SOURCE=wiki \      # options: wiki | google | none
DISC_USE_RAG=1 \
GEN_USE_WIKI=1 \
OUTPUT_DIR=local/runs/demo_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

`train.sh` calls `gan_training.py`. Key arguments:
- `--dataset-name` / `--dataset-config` / `--dataset-split`: HF dataset settings. Default in code is `sanxing/advfake_news_please`; `train.sh` defaults to `cnn_dailymail 3.0.0 train`.
- `--max-samples`: limit sample count for quick runs.
- `--num-rounds`: GAN rounds (generate fakes + train discriminator).
- `--rag-source`: `wiki` | `google` | `none` to choose context source.
- `--disc-use-rag` / `--generator-use-wiki` / `--filter-no-wiki`: whether discriminator gets RAG context, whether generator can fall back to wiki, and whether to skip samples with empty wiki hits.
- `--generator-model` / `--discriminator-model`: HF model IDs.
- `--output-dir`: where generated samples and discriminator checkpoints are written.

## Outputs
When `--output-dir` is set:
```
output_dir/
  round_1/                # HF Dataset with generated samples (text, label, prob_true, suspicious_words, feedback_prompt)
  round_2/
  ...
  disc_round_1/           # per-round discriminator checkpoints (transformers format)
  disc_best/              # best-loss discriminator
```
Inspect quickly:
```bash
python - <<'PY'
from inspect_gan_run import inspect_run
inspect_run("local/runs/demo_20250101_120000", sample_n=3)
PY
```
(Or edit `inspect_gan_run.py` `__main__` and set `run_dir`.)

## Pipeline Overview
1) Load data (`load_dataset` or `load_from_disk`) with fields `title` / `description` or `text`. CNN/DailyMail auto-derives `title` and date.
2) Fetch context by `rag_source`: Wikipedia (default), SerpAPI Google, or none.
3) Generator produces fake news, keeping only `Title/Body` for the discriminator to avoid feeding prompts.
4) Discriminator returns plausibility and suspicious words; feedback is stored for the next round.
5) Mix real/fake samples, train discriminator, and save stats/checkpoints.

## Notes
- VRAM/compute depends on generator size; swap `generator_model` to a smaller HF model (e.g., `gpt2`) if resources are tight.
- Requirements default to PyTorch 2.5.1 + cu121 with `safetensors` to avoid older `torch.load` restrictions. Adjust wheels if you use a different CUDA.
- Wikipedia API needs network access; if blocked, `generator.py` falls back to stub text.
- Respect SerpAPI terms when using Google RAG; do not misuse real search results.
- Research/defense use only; do not deploy for real misinformation.
