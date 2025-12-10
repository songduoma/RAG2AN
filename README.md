# RAG²AN: Retrieval-Augmented Generator & Discriminator for Fake News GAN

> Research/defense use only. Do not deploy this to spread misinformation.

RAG²AN links a retrieval-augmented generator with a DeBERTa-based discriminator in a GAN-style loop and adds **Verbal Adversarial Feedback (VAF)**. The discriminator returns P(real), suspicious tokens, detection reasons, and improvement tips; the generator rewrites with that feedback and can run brief LoRA mini-SFT between rounds. Both local HF (GPU) and OpenAI-compatible API generation are supported. Wikipedia and Google retrieval are available; wiki snippets are re-ranked with a sentence embedding model.

## Highlights
- **VAF feedback loop**: discriminator surfaces suspicious terms, confidence, reasons, and fix suggestions; generator rewrites against that signal.
- **Dynamic balance**: skip discriminator training when fool rate is too low; label smoothing slows D; forced resume after configured skips.
- **Few-shot cache**: fakes that fooled D are stored (up to 5) and injected as hints in later prompts.
- **LoRA mini-SFT**: in local mode, high-score fakes fine-tune LoRA for a few steps with KL regularization and gradient clipping; warm-up SFT allowed.
- **Retrieval choices**: wiki or Google (SerpAPI) for generator and discriminator; wiki snippets ranked via `BAAI/bge-small-en-v1.5` on CPU/CUDA/auto.
- **Data handling**: CNN/DailyMail keeps the shortest 10k articles for speed; date fields normalized; description filled from text when missing.

## Requirements
- Python 3.10+.
- GPU with CUDA 12.1 for local generation; API mode needs no GPU.
- `OPENAI_API_KEY` for OpenAI-compatible models (`OPENAI_BASE_URL` optional).
- `SERPAPI_API_KEY` for Google retrieval.
- Set `HF_TOKEN` if your HF model is gated.

Install:
```bash
cd RAG2AN
pip install -r requirements.txt
```
If your CUDA version differs from 12.1, install matching torch/vision/audio wheels.

## Quickstart
- **Default local run** (Qwen3-4B-Instruct + LoRA, wiki RAG, 20 rounds, CNN/DailyMail shortest 10k):
  ```bash
  ./train.sh
  ```
- **API mode (no GPU)**:
  ```bash
  GEN_MODE=api \
  OPENAI_API_KEY=sk-xxx \
  DATASET_SPLIT=train[:200] \
  NUM_ROUNDS=3 \
  OUTPUT_DIR=local/runs/api_demo_$(date +%Y%m%d_%H%M%S) \
  ./train.sh
  ```
- **Quick smoke test** (tiny split, fewer epochs):
  ```bash
  NUM_ROUNDS=2 \
  DATASET_SPLIT=train[:64] \
  DISC_EPOCHS=1 \
  BATCH_SIZE=2 \
  GEN_SFT_EVERY_ROUND=0 \
  OUTPUT_DIR=local/runs/quick_$(date +%H%M%S) \
  ./train.sh
  ```

## Key settings (env vars read by `train.sh`)
- **Data**: `DATASET_NAME` (default `cnn_dailymail`), `DATASET_CONFIG` (e.g., `3.0.0`), `DATASET_SPLIT`, `REAL_SAMPLES_PER_ROUND` (defaults to the split size). CNN/DailyMail keeps the **shortest 10k** for speed.
- **Generator**: `GEN_MODE`=`local|api`; local uses `GEN_MODEL` (default `Qwen/Qwen3-4B-Instruct-2507`); `GEN_USE_LORA`; `GEN_MAX_NEW_TOKENS`. API uses `OPENAI_MODEL` (default `gpt-4o-mini`) and `OPENAI_API_KEY`.
- **LoRA mini-SFT**: `GEN_SFT_EVERY_ROUND`, `GEN_SFT_LR`, `GEN_SFT_STEPS`, `GEN_SFT_BATCH_SIZE`, `GEN_SFT_MAX_LENGTH`, `GEN_SFT_KL_WEIGHT`, `GEN_SFT_MAX_SAMPLES`, `GEN_SFT_SUCCESS_THRESHOLD`, `GEN_SFT_MAX_GRAD_NORM`, `GEN_SFT_WARMUP_ROUNDS`, `GEN_LORA_R/ALPHA/DROPOUT`.
- **Discriminator**: `DISC_MODEL` (default `microsoft/deberta-v3-base`), `DISC_EPOCHS`, `BATCH_SIZE`, `LR`, `MAX_LENGTH`.
- **Retrieval**: `RAG_SOURCE`=`wiki|google|none`, `DISC_USE_RAG`, `GEN_USE_WIKI` (allow wiki fallback when context is empty), `NUM_RAG_RESULTS`, `RAG_LANG`, `FILTER_NO_WIKI` (skip samples if wiki returns empty), `RAG_EMBED_MODEL`, `RAG_EMBED_QUERY_PREFIX`, `RAG_EMBED_DEVICE` (`cpu|cuda|auto`).
- **Dynamic balance**: `LABEL_SMOOTHING` (default 0.1), `MIN_FOOL_RATE` (skip D training below this), `MAX_SKIP_ROUNDS`.
- **Output/logging**: `OUTPUT_DIR` (default `local/rag_gan_runs/<timestamp>`), `LOG_INTERVAL`.

## Direct script call
```bash
python -u gan_training.py \
  --dataset-name sanxing/advfake_news_please \
  --dataset-split train[:1000] \
  --num-rounds 2 \
  --rag-source wiki \
  --generator-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir local/runs/manual_call
```
Use `--disc-use-rag/--no-disc-rag` and `--generator-use-wiki/--no-generator-wiki` to toggle retrieval; other CLI flags mirror the env vars above.

## Outputs
When `OUTPUT_DIR` is set:
```
output_dir/
  training_history.json   # per-round stats: mean P(real), fool rate, loss, etc.
  round_1/                # fake samples + feedback (HF dataset)
  round_2/
  disc_round_1/           # discriminator checkpoints (per round)
  disc_best/              # best-loss discriminator
```
Each round prints sample inspections, suspicious term distribution, fool rate, and whether D training was skipped.

## Generator demo
```bash
# Local mode needs GPU; API mode needs OPENAI_API_KEY
python generator.py
```
Programmatic use:
```python
from generator import generator

fake = generator(
    title="Tech Firm Unveils Antarctic Campus",
    content="A major tech firm announced ...",
    feedback_prompt="Avoid words like 'shocking'.",
    use_rag=True,
    rag_query="Technology companies in Antarctica",
    num_rag_results=3,
)
print(fake)
```

## Data prep notes
- For `cnn_dailymail`, the shortest quarter (up to 10k) of articles is kept; titles use highlights or a truncated article.
- Other datasets try to fill `description` from `text` and normalize `date_publish`.
- `--filter-no-wiki` can skip samples when wiki RAG returns empty context to avoid blank prompts.

## Tips and caveats
- Wiki/Google retrieval needs network; if blocked, a stub context is returned.
- GPU memory depends on `GEN_MODEL`; choose a smaller model or API mode if constrained.
- `GEN_SFT_EVERY_ROUND=1` only affects local + LoRA; API mode does not fine-tune.
- Defense/research only—do not deploy generated content in real information channels.
