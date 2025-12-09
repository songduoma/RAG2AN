#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to launch the RAG-GAN training loop.
# Adjust the variables below to fit your environment / resources.

# ============================================================
# VERBAL ADVERSARIAL FEEDBACK (VAF) - Enhanced GAN Training
# ============================================================

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF

# --- Generator 模式設定 ---
# 'api' = 使用 OpenAI API（推薦，不需 GPU）
# 'local' = 使用本地模型（需要 GPU）
GEN_MODE="${GEN_MODE:-local}"

# API 設定（GEN_MODE=api 時需要）
# export OPENAI_API_KEY=sk-xxx  # 在執行前設定
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

# 本地模型設定（GEN_MODE=local 時使用）
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
GEN_USE_LORA="${GEN_USE_LORA:-1}"
GEN_SFT_EVERY_ROUND="${GEN_SFT_EVERY_ROUND:-1}"
GEN_SFT_LR="${GEN_SFT_LR:-1e-4}"
GEN_SFT_STEPS="${GEN_SFT_STEPS:-3}"
GEN_SFT_BATCH_SIZE="${GEN_SFT_BATCH_SIZE:-1}"
GEN_SFT_MAX_LENGTH="${GEN_SFT_MAX_LENGTH:-512}"
GEN_SFT_KL_WEIGHT="${GEN_SFT_KL_WEIGHT:-0.01}"
GEN_SFT_MAX_SAMPLES="${GEN_SFT_MAX_SAMPLES:-10}"
GEN_SFT_SUCCESS_THRESHOLD="${GEN_SFT_SUCCESS_THRESHOLD:-0.5}"
GEN_SFT_MAX_GRAD_NORM="${GEN_SFT_MAX_GRAD_NORM:-1.0}"
GEN_SFT_WARMUP_ROUNDS="${GEN_SFT_WARMUP_ROUNDS:-10}"
GEN_LORA_R="${GEN_LORA_R:-8}"
GEN_LORA_ALPHA="${GEN_LORA_ALPHA:-16}"
GEN_LORA_DROPOUT="${GEN_LORA_DROPOUT:-0.05}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-512}"

# --- 資料集設定 ---
DATASET_NAME="${DATASET_NAME:-cnn_dailymail}"        # HF dataset id to load (e.g., cnn_dailymail)
DATASET_CONFIG="${DATASET_CONFIG:-3.0.0}"            # optional dataset config/version
DATASET_SPLIT="${DATASET_SPLIT:-train[:40000]}"         # HF split selector (smaller for API cost)
NUM_ROUNDS="${NUM_ROUNDS:-20}"                        # GAN rounds (generate + train disc)

# --- 訓練設定 ---
LOG_INTERVAL="${LOG_INTERVAL:-10}"                    # print progress every N samples
DISC_EPOCHS="${DISC_EPOCHS:-1}"                      # epochs per round for discriminator
BATCH_SIZE="${BATCH_SIZE:-2}"                        # discriminator batch size
LR="${LR:-1e-5}"                                     # discriminator learning rate
MAX_LENGTH="${MAX_LENGTH:-512}"

# --- RAG 設定 ---
RAG_SOURCE="${RAG_SOURCE:-wiki}"                     # retrieval source: google | wiki | none
DISC_USE_RAG="${DISC_USE_RAG:-0}"                    # 1 to include RAG context in discriminator input
GEN_USE_WIKI="${GEN_USE_WIKI:-1}"                    # 1 to let generator fallback to wiki when no context
NUM_RAG_RESULTS="${NUM_RAG_RESULTS:-3}"              # number of RAG hits to fetch
RAG_LANG="${RAG_LANG:-en}"                           # wiki language for RAG

# --- 動態平衡設定（防止 D 壓制 G）---
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"            # Label smoothing (讓 D 學慢一點)
MIN_FOOL_RATE="${MIN_FOOL_RATE:-0.05}"               # G 低於此 fool rate 時暫停訓練 D
MAX_SKIP_ROUNDS="${MAX_SKIP_ROUNDS:-3}"              # 最多連續跳過幾輪 D 訓練

# --- 模型設定 ---
DISC_MODEL="${DISC_MODEL:-microsoft/deberta-v3-base}" # discriminator model id
ENCODER_DISCRIMINATOR_MODEL="${ENCODER_DISCRIMINATOR_MODEL:-$DISC_MODEL}"
ENCODER_DISCRIMINATOR_MAX_LEN="${ENCODER_DISCRIMINATOR_MAX_LEN:-$MAX_LENGTH}"
REAL_SAMPLES_PER_ROUND="${REAL_SAMPLES_PER_ROUND:-500}"

# --- 輸出設定 ---
OUTPUT_DIR="${OUTPUT_DIR:-local/rag_gan_runs/$(date +%Y%m%d_%H%M%S)}"

cd "$(dirname "$0")"

# 設定環境變數
export GEN_MODE="$GEN_MODE"
export OPENAI_MODEL="$OPENAI_MODEL"
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
# OPENAI_API_KEY 只在 api 模式下需要，這裡不覆蓋使用者環境
export GEN_MODEL="$GEN_MODEL"
export GEN_USE_LORA="$GEN_USE_LORA"
export GEN_SFT_EVERY_ROUND="$GEN_SFT_EVERY_ROUND"
export GEN_LORA_R="$GEN_LORA_R"
export GEN_LORA_ALPHA="$GEN_LORA_ALPHA"
export GEN_LORA_DROPOUT="$GEN_LORA_DROPOUT"
export GEN_MAX_NEW_TOKENS="$GEN_MAX_NEW_TOKENS"
export ENCODER_DISCRIMINATOR_MODEL="$ENCODER_DISCRIMINATOR_MODEL"
export ENCODER_DISCRIMINATOR_MAX_LEN="$ENCODER_DISCRIMINATOR_MAX_LEN"

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
  $( [[ -n "$REAL_SAMPLES_PER_ROUND" ]] && echo "--real-samples-per-round $REAL_SAMPLES_PER_ROUND" ) \
  $( [[ "$GEN_SFT_EVERY_ROUND" == "1" ]] && echo "--gen-sft-every-round" || echo "--no-gen-sft" ) \
  --gen-sft-lr "$GEN_SFT_LR" \
  --gen-sft-steps "$GEN_SFT_STEPS" \
  --gen-sft-batch-size "$GEN_SFT_BATCH_SIZE" \
  --gen-sft-max-length "$GEN_SFT_MAX_LENGTH" \
  --gen-sft-success-threshold "${GEN_SFT_SUCCESS_THRESHOLD}" \
  --gen-sft-max-samples "$GEN_SFT_MAX_SAMPLES" \
  --gen-sft-lambda-kl "$GEN_SFT_KL_WEIGHT" \
  --gen-sft-max-grad-norm "${GEN_SFT_MAX_GRAD_NORM}" \
  --gen-sft-warmup-rounds "$GEN_SFT_WARMUP_ROUNDS" \
  --log-interval "$LOG_INTERVAL" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "${MAX_LENGTH}" \
  --label-smoothing "$LABEL_SMOOTHING" \
  --min-fool-rate-to-train "$MIN_FOOL_RATE" \
  --max-skip-rounds "$MAX_SKIP_ROUNDS"

echo "============================================================"
echo "Training run finished. Outputs saved to: $OUTPUT_DIR"
echo "============================================================"
