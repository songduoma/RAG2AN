# RAG²AN: Retrieval-Augmented Generator & Discriminator for Fake News GAN

> Research/defense use only. Do not deploy this to spread misinformation.

RAG²AN links a retrieval-augmented generator with a DeBERTa-based discriminator inside a GAN-style loop and drives the generator with **Verbal Adversarial Feedback (VAF)**. The discriminator returns P(real), suspicious tokens, detection reasons, and rewrite advice; the generator incorporates that feedback and can run short LoRA mini-SFT bursts between rounds. Generation supports both OpenAI-compatible APIs (no GPU) and local HF models. Retrieval uses a local DPR index built from `sanxing/advfake_news_please`.

## Highlights
- **VAF feedback loop**: discriminator explains why a draft looked fake and lists suspicious spans plus concrete rewrite tips that are appended to the next prompt.
- **Dynamic balance**: label smoothing, fool-rate-triggered pauses, and forced resumes keep the discriminator from overpowering the generator.
- **Few-shot cache**: up to five fake articles that fooled the discriminator are stored and replayed in future prompts.
- **LoRA mini-SFT**: in local mode high-score fakes fine-tune the generator (KL-regularized, gradient-clipped, warm-up enabled) using LoRA adapters only.
- **Local DPR retrieval**: `src/retrieval_dpr.py` builds and caches a FAISS index under `local/news-please/faiss_index` on first use; both generator and discriminator can inject that context.
- **Data hygiene**: `scripts/gan_training.py` filters `sanxing/advfake_news_please`, drops very short boilerplate, and removes the 402 real descriptions from `sanxing/advfake` to avoid leakage.

## Requirements & setup
- Python 3.10+.
- GPU with CUDA 12.1 if you want to run the local generator; discriminator/evaluator work on CPU, and API generation needs no GPU.
- `OPENAI_API_KEY` for API mode and `HF_TOKEN` for gated local checkpoints.

Install dependencies:
```bash
pip install -r requirements.txt
```
If your CUDA toolchain differs from the pinned `torch==2.6.0+cu118`, install compatible wheels before running `pip install -r requirements.txt`.

### Retrieval cache
The first call to `retrieval_dpr.get_dpr()` downloads `sanxing/advfake_news_please`, builds DPR embeddings, and saves a FAISS index at `local/news-please/faiss_index/my_index.faiss`. Expect a few minutes of preprocessing; subsequent runs reuse the cache automatically.

## Quickstart
### Default training run (local mode)
```bash
./train.sh
```
Defaults: `sanxing/advfake_news_please` split `train[:10000]`, `NUM_ROUNDS=10`, generator `Qwen/Qwen3-4B-Instruct-2507` with LoRA mini-SFT every round, discriminator `microsoft/deberta-v3-base`, DPR retrieval disabled for the discriminator but available to the generator via `GEN_USE_RAG`.

### API-only generation
```bash
GEN_MODE=api \
OPENAI_API_KEY=sk-xxx \
OPENAI_MODEL=gpt-4o-mini \
DATASET_SPLIT=train[:200] \
NUM_ROUNDS=3 \
OUTPUT_DIR=local/runs/api_demo_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

### Smoke test
```bash
NUM_ROUNDS=2 \
DATASET_SPLIT=train[:64] \
DISC_EPOCHS=1 \
BATCH_SIZE=2 \
GEN_SFT_EVERY_ROUND=0 \
OUTPUT_DIR=local/runs/quick_$(date +%H%M%S) \
./train.sh
```

## Key environment variables (`train.sh`)
- **Data**: `DATASET_NAME` (default `sanxing/advfake_news_please`), `DATASET_SPLIT`, `NUM_ROUNDS`, `REAL_SAMPLES_PER_ROUND`.
- **Generator**: `GEN_MODE` (`local|api`), `GEN_MODEL`, `GEN_USE_LORA`, `GEN_MAX_NEW_TOKENS`, `GEN_USE_RAG`.
- **LoRA mini-SFT**: `GEN_SFT_EVERY_ROUND`, `GEN_SFT_LR`, `GEN_SFT_STEPS`, `GEN_SFT_BATCH_SIZE`, `GEN_SFT_MAX_LENGTH`, `GEN_SFT_MAX_SAMPLES`, `GEN_SFT_SUCCESS_THRESHOLD`, `GEN_SFT_MAX_GRAD_NORM`, `GEN_SFT_WARMUP_ROUNDS`, `GEN_SFT_KL_WEIGHT`, `GEN_LORA_R`, `GEN_LORA_ALPHA`, `GEN_LORA_DROPOUT`.
- **Discriminator**: `DISC_MODEL`, `DISC_EPOCHS`, `BATCH_SIZE`, `LR`, `MAX_LENGTH`.
- **Retrieval**: `RAG_SOURCE` (`dpr|none`), `DISC_USE_RAG`, `GEN_USE_RAG`, `NUM_RAG_RESULTS`.
- **Dynamic balance**: `LABEL_SMOOTHING`, `MIN_FOOL_RATE`, `MAX_SKIP_ROUNDS`.
- **Verbal Adversarial Feedback**: `USE_VAF_FEEDBACK` (insert discriminator feedback block), `USE_VAF_FEWSHOT` (insert successful example few-shot block). Legacy `USE_VAF` sets both at once.
- **Output/logging**: `OUTPUT_DIR` (default `local/rag_gan_runs/<timestamp>`), `LOG_INTERVAL`, `LOG_FILE`.

## Direct script call
Every train.sh flag maps to `scripts/gan_training.py` CLI arguments:
```bash
python -u scripts/gan_training.py \
  --dataset-name sanxing/advfake_news_please \
  --dataset-split train[:1000] \
  --num-rounds 2 \
  --rag-source dpr \
  --gen-use-rag \
  --generator-model Qwen/Qwen3-4B-Instruct-2507 \
  --discriminator-model microsoft/deberta-v3-base \
  --output-dir local/runs/manual_call
```
Use `--disc-use-rag/--no-disc-rag` and `--gen-use-rag/--no-gen-rag` to toggle retrieval per component.

## Outputs
When `OUTPUT_DIR` is set:
```
output_dir/
  training_history.json   # JSON summary per round
  round_1/                # HF dataset with generated fake samples + metadata
  ...
  disc_round_1/           # discriminator checkpoint for the round
  disc_best/              # best-loss discriminator snapshot
  train.log               # tee'd stdout/stderr if LOG_FILE points here
```
Round logs cover sample inspections, mean P(real), fool rate, detection reasons, and whether discriminator training was skipped.

## Evaluating discriminators
Two scripts score saved checkpoints on `sanxing/advfake`:

1. **Plain inputs**:
   ```bash
   python scripts/evaluate_discriminator.py \
     --models-dir local/rag_gan_runs/<run_name> \
     --dataset-path local/hf_datasets/advfake/advfake-train.arrow \
     --batch-size 8
   ```
   The arrow file can be created via `load_dataset("sanxing/advfake", split="train").with_format("arrow").save_to_disk(...)`, or grab it from an existing HF cache.

2. **RAG-augmented inputs**:
   ```bash
   python scripts/evaluate_discriminator_with_rag.py \
     --models-dir local/rag_gan_runs/<run_name> \
     --dataset-path local/hf_datasets/advfake/advfake-train.arrow \
     --rag-source dpr \
     --num-rag-results 3 \
     --batch-size 4
   ```
   This version recreates the exact DPR prompt structure (with caching under `local/rag_cache`).

The scripts print Macro-F1 / ROC-AUC / Accuracy per checkpoint plus a round-by-round summary.

## Generator demo
```bash
# Local mode needs GPU; API mode needs OPENAI_API_KEY
python -m src.generator
```
Programmatic use:
```python
from src.generator import FakeNewsGenerator

gen = FakeNewsGenerator()
sample = gen.generate(
    content="A major tech firm announced ...",
    feedback_prompt="Avoid words like 'shocking'.",
    use_rag=True,
    rag_query="Technology companies in Antarctica",
    num_rag_results=3,
)
print(sample)
```

## Data notes
- `load_news_data` keeps only descriptions longer than 50 chars and drops any entry that matches the 402 real descriptions from `sanxing/advfake` to prevent contamination.
- Dataset slices (e.g., `train[:1000]`) determine how many unique real articles feed each round; `REAL_SAMPLES_PER_ROUND` can cap the per-round batch.
- Retrieval context always comes from DPR (no Google/SerpAPI usage in this codebase).

## Tips & caveats
- API mode skips LoRA fine-tuning; set `GEN_MODE=local` to unlock SFT and ensure your GPU has enough VRAM for the chosen model.
- The first DPR indexing pass loads the full dataset on GPU; keep `local/news-please` on a fast disk.
- Set `GEN_MAX_NEW_TOKENS` conservatively—the generator already scales length to the source article.
- Defense/research only—do not deploy generated content in real information channels.
