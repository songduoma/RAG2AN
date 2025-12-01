# RAG²AN – Retrieval-Augmented Generator & Discriminator for Fake News GAN

> Research use only. This code is for fake-news detection and adversarial generation research. Do not use it to spread misinformation.

RAG²AN ties a generator with retrieval-augmented generation (RAG) and an encoder-based discriminator into a simple GAN-style loop. The generator crafts suspect news with external context; the discriminator judges plausibility and surfaces suspicious words for feedback, iteratively improving the discriminator.

## Project Structure
- `gan_training.py`: Main training loop for loading data, calling the generator, training the discriminator, and saving checkpoints.
- `train.sh`: Convenience launcher; tweak environment variables to change dataset, rounds, and models.
- `generator.py`: Default generator is `Qwen/Qwen2.5-7B-Instruct`, with built-in Wikipedia RAG and feedback prompt support.
- `discriminator.py`: Encoder-based discriminator (DeBERTa) that can return suspicious words via attention.
- `search.py`: Google Search RAG via SerpAPI.

## Requirements & Install
- Python 3.10 with CUDA 12.1 GPU.
- HF token: Accept Qwen 7B terms on Hugging Face and set `HF_TOKEN`.
- Google RAG: set `SERPAPI_API_KEY` (package `google-search-results` is included).

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

## Generator Demo & API
`generator.py` now ships with a simple demo that loads the first five records from `cnn_dailymail` (`test[:5]`), gathers Wikipedia snippets for each prompt, and feeds them into the `Qwen/Qwen2.5-7B-Instruct` pipeline. Running `python generator.py` prints a comparison report that labels the entire original article as **REAL NEWS (Full Text)** followed by the **GENERATED FAKE NEWS** version, making it easy to review how the GAN-textifier alters the source.

Behind the scenes, the demo uses `search_wikipedia` → `LlamaEngine` → `generate_fake_news` so that every fake article keeps the original paragraph layout, avoids markdown, and begins directly with the news content (e.g., “(CNN)...”).

### Programmatic usage
Import `FakeNewsGenerator` or call the exported helper `generator()` to reuse the pipeline in other scripts. Example:
```python
from generator import generator

fake = generator(
    title="Tech Firm Unveils Antarctic Campus",
    content="A major tech firm announced a plan ...",
    feedback_prompt="Avoid words like 'shocking' or 'anonymous sources'.",
    use_rag=True,
    rag_query="Technology companies in Antarctica",
    lang="en",
    num_rag_results=3
)
print(fake)
```

Key arguments:
- `feedback_prompt`: textual guidance to steer the generator away from obviously fake phrasing.
- `use_rag` / `rag_query` / `context_override`: controls whether and how Wikipedia context is pulled; `context_override` bypasses the lookup.
- `lang`, `num_rag_results`: configure the wiki locale and number of snippets per query.
- `model_id`: override the default `MODEL_ID` (`Qwen/Qwen2.5-7B-Instruct`) if you want a smaller or licensed model.

The helper lazily instantiates `FakeNewsGenerator`, which wraps `LlamaEngine` and the `search_wikipedia` retrieval logic. When the wiki endpoint fails, `search_wikipedia` returns a lightweight fallback snippet so the demo keeps running.

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
    state.json            # per-round stats produced by `gan_training.py` (avg loss, sample counts, etc.)
  round_2/
  ...
  disc_round_1/           # per-round discriminator checkpoints (transformers format)
  disc_best/              # best-loss discriminator
```


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
