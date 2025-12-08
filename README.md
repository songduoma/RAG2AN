# RAG²AN – Retrieval-Augmented Generator & Discriminator for Fake News GAN

> Research use only. This code is for fake-news detection and adversarial generation research. Do not use it to spread misinformation.

RAG²AN couples a retrieval-augmented generator and a DeBERTa-style discriminator into a GAN-like loop. The latest code adds **Verbal Adversarial Feedback (VAF)**: the discriminator surfaces suspicious tokens plus structured reasons and suggestions; the generator adapts with that feedback and optional LoRA mini-SFT between rounds. Training can run with a local HF model or an OpenAI-compatible API model.

## Requirements & Install
- Python 3.10+, CUDA 12.1 GPU for local mode (`GEN_MODE=local`). API mode (`GEN_MODE=api`) needs no GPU.
- Hugging Face access if your chosen HF model is gated (export `HF_TOKEN`).
- Google RAG requires `SERPAPI_API_KEY`.
- OpenAI-compatible generation requires `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`).

Install:
```bash
cd RAG2AN
# activate your venv/conda env first if needed
pip install -r requirements.txt   # includes torch/vision/audio cu121 wheels
```
> On non-CUDA-12.1 systems, edit the `--extra-index-url` or install torch/torchvision/torchaudio that match your CUDA/CPU.

## Quickstart
- **Local (default)**: Qwen3 4B Instruct + LoRA, Wiki RAG, 20 rounds, CNN/DailyMail slice.
  ```bash
  ./train.sh
  ```
  Defaults in `train.sh`: `DATASET_NAME=cnn_dailymail`, `DATASET_CONFIG=3.0.0`, `DATASET_SPLIT=train[:2000]`, `NUM_ROUNDS=20`, `GEN_MODEL=Qwen/Qwen3-4B-Instruct-2507`, `GEN_MODE=local`.

- **API mode (no GPU)**:
  ```bash
  GEN_MODE=api \
  OPENAI_API_KEY=sk-xxx \
  DATASET_SPLIT=train[:100] \
  NUM_ROUNDS=3 \
  OUTPUT_DIR=local/runs/api_demo_$(date +%Y%m%d_%H%M%S) \
  ./train.sh
  ```

- **Minimal local run** (faster smoke test, smaller batches):
  ```bash
  NUM_ROUNDS=2 \
  DATASET_SPLIT=train[:64] \
  DISC_EPOCHS=1 \
  BATCH_SIZE=2 \
  GEN_SFT_EVERY_ROUND=0 \
  OUTPUT_DIR=local/runs/quick_$(date +%H%M%S) \
  ./train.sh
  ```

`train.sh` is a thin wrapper over `gan_training.py`; you can also call it directly, e.g.:
```bash
python -u gan_training.py \
  --dataset-name sanxing/advfake_news_please \
  --dataset-split train \
  --num-rounds 2 \
  --rag-source wiki \
  --generator-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir local/runs/manual_call
```

## Key knobs (env vars consumed by `train.sh`)
- **Generator**: `GEN_MODE` (`local`|`api`), `GEN_MODEL` (HF id for local), `OPENAI_MODEL` (API id), `GEN_USE_LORA` (1/0), `GEN_SFT_EVERY_ROUND`, `GEN_SFT_LR`, `GEN_SFT_STEPS`, `GEN_SFT_BATCH_SIZE`, `GEN_SFT_MAX_LENGTH`, `GEN_SFT_SUCCESS_THRESHOLD`, `GEN_SFT_MAX_SAMPLES`, `GEN_SFT_WARMUP_ROUNDS`, `GEN_LORA_R`, `GEN_LORA_ALPHA`, `GEN_LORA_DROPOUT`.
- **Data**: `DATASET_NAME`, `DATASET_CONFIG`, `DATASET_SPLIT` (Hugging Face slice), `REAL_SAMPLES_PER_ROUND` (defaults to 50 in the wrapper).
- **Discriminator**: `DISC_MODEL`, `DISC_EPOCHS`, `BATCH_SIZE`, `LR`, `MAX_LENGTH`, `LABEL_SMOOTHING`.
- **RAG & balance**: `RAG_SOURCE` (`wiki`|`google`|`none`), `DISC_USE_RAG` (1/0), `GEN_USE_WIKI` (1/0 fallback), `NUM_RAG_RESULTS`, `RAG_LANG`, `MIN_FOOL_RATE` / `MAX_SKIP_ROUNDS` (dynamic pause of D when G is weak).
- **Output/logging**: `OUTPUT_DIR`, `LOG_INTERVAL`.

## Outputs
When `--output-dir` / `OUTPUT_DIR` is set:
```
output_dir/
  training_history.json   # per-round stats (mean P(real), fool rate, loss, etc.)
  round_1/                # HF dataset with fake samples + feedback metadata
  round_2/
  ...
  disc_round_1/           # discriminator checkpoints (per round)
  disc_best/              # best-loss discriminator
```

## Generator demo & API
- Mode is controlled by `GEN_MODE` (`local` uses HF weights + optional LoRA; `api` uses OpenAI-compatible chat completions).
- Run `python generator.py` to see a small two-step demo (with and without feedback). It will auto-pull Wikipedia snippets unless `use_rag=False`.

Programmatic helper:
```python
from generator import generator

fake = generator(
    title="Tech Firm Unveils Antarctic Campus",
    content="A major tech firm announced a plan ...",
    feedback_prompt="Avoid words like 'shocking' or 'anonymous sources'.",
    use_rag=True,
    rag_query="Technology companies in Antarctica",
    lang="en",
    num_rag_results=3,
)
print(fake)
```
Pass `model_id=` if you want to override the HF model in local mode.

## Pipeline overview
- Data: loads HF datasets (`load_dataset` or `load_from_disk`). For `cnn_dailymail`, the longer half of articles is dropped for speed; titles derive from highlights.
- RAG: `RAG_SOURCE=wiki` (default) hits Wikipedia; `google` uses SerpAPI; `none` skips retrieval. `GEN_USE_WIKI=1` lets the generator fall back to wiki when the context is empty.
- Generation: rewrites real news to fake, enforces “no markdown, keep paragraph structure”. Feedback from prior rounds is injected into the prompt. In local mode, LoRA adapters can be fine-tuned briefly each round on successful fakes.
- Discrimination: DeBERTa classifier returns P(real), suspicious words, detection reasons, and improvement suggestions. Label smoothing plus dynamic skip logic prevent the discriminator from overpowering the generator.

## Notes
- VRAM depends on the local generator; swap `GEN_MODEL` to a smaller HF model if resources are tight. API mode avoids GPU needs but costs tokens.
- Wikipedia API and SerpAPI require network access; if blocked, wiki retrieval falls back to stub text.
- Research/defense use only; do not deploy for real misinformation.
