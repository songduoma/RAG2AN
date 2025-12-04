#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to launch the RAG-GAN training loop.
# Adjust the variables below to fit your environment / resources.

# ============================================================
# VERBAL ADVERSARIAL FEEDBACK (VAF) - Enhanced GAN Training
# ============================================================

# --- Generator 模式設定 ---
# 'api' = 使用 OpenAI API（推薦，不需 GPU）
# 'local' = 使用本地模型（需要 GPU）
GEN_MODE="${GEN_MODE:-api}"

# API 設定（GEN_MODE=api 時需要）
# export OPENAI_API_KEY=sk-xxx  # 在執行前設定
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

# 本地模型設定（GEN_MODE=local 時使用）
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# --- 資料集設定 ---
DATASET_NAME="${DATASET_NAME:-cnn_dailymail}"        # HF dataset id to load (e.g., cnn_dailymail)
DATASET_CONFIG="${DATASET_CONFIG:-3.0.0}"            # optional dataset config/version
DATASET_SPLIT="${DATASET_SPLIT:-train[:50]}"         # HF split selector (smaller for API cost)
NUM_ROUNDS="${NUM_ROUNDS:-5}"                        # GAN rounds (generate + train disc)

# --- 訓練設定 ---
LOG_INTERVAL="${LOG_INTERVAL:-5}"                    # print progress every N samples
DISC_EPOCHS="${DISC_EPOCHS:-1}"                      # epochs per round for discriminator
BATCH_SIZE="${BATCH_SIZE:-4}"                        # discriminator batch size
LR="${LR:-2e-5}"                                     # discriminator learning rate

# --- RAG 設定 ---
RAG_SOURCE="${RAG_SOURCE:-wiki}"                     # retrieval source: google | wiki | none
DISC_USE_RAG="${DISC_USE_RAG:-1}"                    # 1 to include RAG context in discriminator input
GEN_USE_WIKI="${GEN_USE_WIKI:-1}"                    # 1 to let generator fallback to wiki when no context
NUM_RAG_RESULTS="${NUM_RAG_RESULTS:-3}"              # number of RAG hits to fetch
RAG_LANG="${RAG_LANG:-en}"                           # wiki language for RAG

# --- 動態平衡設定（防止 D 壓制 G）---
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"            # Label smoothing (讓 D 學慢一點)
MIN_FOOL_RATE="${MIN_FOOL_RATE:-0.05}"               # G 低於此 fool rate 時暫停訓練 D
MAX_SKIP_ROUNDS="${MAX_SKIP_ROUNDS:-2}"              # 最多連續跳過幾輪 D 訓練

# --- 模型設定 ---
DISC_MODEL="${DISC_MODEL:-microsoft/deberta-v3-base}" # discriminator model id

# --- 輸出設定 ---
OUTPUT_DIR="${OUTPUT_DIR:-local/rag_gan_runs/$(date +%Y%m%d_%H%M%S)}"

cd "$(dirname "$0")"

# 設定環境變數
export GEN_MODE="$GEN_MODE"
export OPENAI_MODEL="$OPENAI_MODEL"
export GEN_MODEL="$GEN_MODEL"

echo "============================================================"
echo "VERBAL ADVERSARIAL FEEDBACK (VAF) TRAINING"
echo "============================================================"
echo "Generator Mode: $GEN_MODE"
if [[ "$GEN_MODE" == "api" ]]; then
    echo "  API Model:   $OPENAI_MODEL"
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "  ⚠️  WARNING: OPENAI_API_KEY not set!"
        echo "  Run: export OPENAI_API_KEY=sk-xxx"
        exit 1
    fi
else
    echo "  Local Model: $GEN_MODEL"
fi
echo "Discriminator: $DISC_MODEL"
echo "Dataset:       $DATASET_NAME ($DATASET_SPLIT)"
echo "Rounds:        $NUM_ROUNDS"
echo "RAG Source:    $RAG_SOURCE (Disc RAG: $DISC_USE_RAG)"
echo "Output:        $OUTPUT_DIR"
echo "============================================================"

python -u gan_training.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-config "$DATASET_CONFIG" \
  --dataset-split "$DATASET_SPLIT" \
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
  --log-interval "$LOG_INTERVAL" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "${MAX_LENGTH:-384}" \
  --label-smoothing "$LABEL_SMOOTHING" \
  --min-fool-rate-to-train "$MIN_FOOL_RATE" \
  --max-skip-rounds "$MAX_SKIP_ROUNDS"

echo "============================================================"
echo "Training run finished. Outputs saved to: $OUTPUT_DIR"
echo "============================================================"