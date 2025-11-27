#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to launch the RAG-GAN training loop.
# Adjust the variables below to fit your environment / resources.

DATASET_NAME="${DATASET_NAME:-cnn_dailymail}"        # HF dataset id to load (e.g., cnn_dailymail)
DATASET_CONFIG="${DATASET_CONFIG:-3.0.0}"            # optional dataset config/version
DATASET_SPLIT="${DATASET_SPLIT:-train}"              # HF split selector
MAX_SAMPLES="${MAX_SAMPLES:-20}"                     # cap examples for a quick run
NUM_ROUNDS="${NUM_ROUNDS:-10}"                       # GAN rounds (generate + train disc)
DISC_EPOCHS="${DISC_EPOCHS:-1}"                      # epochs per round for discriminator
BATCH_SIZE="${BATCH_SIZE:-4}"                        # discriminator batch size
LR="${LR:-3e-5}"                                     # discriminator learning rate
RAG_SOURCE="${RAG_SOURCE:-wiki}"                     # retrieval source: google | wiki | none
DISC_USE_RAG="${DISC_USE_RAG:-0}"                    # 1 to include RAG context in discriminator input
GEN_USE_WIKI="${GEN_USE_WIKI:-1}"                    # 1 to let generator fallback to wiki when no context
NUM_RAG_RESULTS="${NUM_RAG_RESULTS:-3}"              # number of RAG hits to fetch
RAG_LANG="${RAG_LANG:-en}"                           # wiki language for RAG
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"   # generator model id
DISC_MODEL="${DISC_MODEL:-microsoft/deberta-v3-base}" # discriminator model id
OUTPUT_DIR="${OUTPUT_DIR:-local/rag_gan_runs/$(date +%Y%m%d_%H%M%S)}"  # where to save each run

cd "$(dirname "$0")"

python -u gan_training.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --dataset-split "$DATASET_SPLIT" \
  --max-samples "$MAX_SAMPLES" \
  --num-rounds "$NUM_ROUNDS" \
  --discriminator-epochs "$DISC_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --rag-source "$RAG_SOURCE" \
  $( [[ "$DISC_USE_RAG" == "1" ]] && echo "--disc-use-rag" || echo "--no-disc-rag" ) \
  $( [[ "$GEN_USE_WIKI" == "1" ]] && echo "--generator-use-wiki" || echo "--no-generator-wiki" ) \
  --num-rag-results "$NUM_RAG_RESULTS" \
  --rag-lang "$RAG_LANG" \
  --generator-model "$GEN_MODEL" \
  --discriminator-model "$DISC_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "${MAX_LENGTH:-384}"

echo "Training run finished. Outputs saved to: $OUTPUT_DIR"
