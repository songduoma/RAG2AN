#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RAG²AN with LoRA - 完整測試流程
# ============================================================
# 
# 此腳本展示如何：
# 1. 使用原始模型進行 GAN 訓練收集資料
# 2. 使用收集的資料訓練 LoRA adapter
# 3. 使用 LoRA 模型繼續 GAN 訓練
# ============================================================

echo "============================================================"
echo "RAG²AN with LoRA - Integration Test"
echo "============================================================"
echo ""

# 設定 GPU 和 NCCL（針對 RTX 4000 系列）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# 建立輸出目錄
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="./test_lora_integration_${TIMESTAMP}"
mkdir -p "$BASE_DIR"

echo "[Config]"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Base Dir: $BASE_DIR"
echo "  NCCL P2P: Disabled"
echo "  NCCL IB: Disabled"
echo ""

# ============================================================
# Phase 1: 使用原始模型進行初始 GAN 訓練
# ============================================================

echo "============================================================"
echo "PHASE 1: Initial GAN Training (Collect Data)"
echo "============================================================"
echo "Training with vanilla model to collect successful samples..."
echo ""

export GEN_MODE=local
export GEN_MODEL="Qwen/Qwen2.5-7B-Instruct"

PHASE1_OUTPUT="${BASE_DIR}/phase1_vanilla_gan"

# 小規模訓練：5 個樣本，3 輪
DATASET_SPLIT="train[:5]" \
NUM_ROUNDS=3 \
BATCH_SIZE=2 \
DISC_EPOCHS=1 \
OUTPUT_DIR="$PHASE1_OUTPUT" \
./train.sh

echo ""
echo "Phase 1 complete. Output: $PHASE1_OUTPUT"
echo ""

# 檢查是否有成功的樣本
SUCCESSFUL_SAMPLES=$(find "$PHASE1_OUTPUT" -name "round_*" -type d | wc -l)
echo "Collected data from $SUCCESSFUL_SAMPLES rounds"

if [[ $SUCCESSFUL_SAMPLES -lt 1 ]]; then
    echo "ERROR: No training data collected. Exiting."
    exit 1
fi

# ============================================================
# Phase 2: 訓練 LoRA Adapter
# ============================================================

echo ""
echo "============================================================"
echo "PHASE 2: LoRA Training"
echo "============================================================"
echo "Training LoRA adapter from collected samples..."
echo ""

LORA_OUTPUT="${BASE_DIR}/lora_adapter"

python -u lora_trainer.py \
    --gan-outputs "$PHASE1_OUTPUT" \
    --output-dir "$LORA_OUTPUT" \
    --model-id "Qwen/Qwen2.5-7B-Instruct" \
    --epochs 2 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --lr 2e-4 \
    --use-4bit

echo ""
echo "Phase 2 complete. LoRA adapter: $LORA_OUTPUT"
echo ""

# ============================================================
# Phase 3: 使用 LoRA 模型繼續 GAN 訓練
# ============================================================

echo ""
echo "============================================================"
echo "PHASE 3: GAN Training with LoRA"
echo "============================================================"
echo "Continuing GAN training with LoRA-enhanced generator..."
echo ""

export GEN_MODE=lora
export LORA_PATH="$LORA_OUTPUT"
export USE_4BIT=1

PHASE3_OUTPUT="${BASE_DIR}/phase3_lora_gan"

# 再訓練 3 輪，看 LoRA 是否提升效果
DATASET_SPLIT="train[:5]" \
NUM_ROUNDS=3 \
BATCH_SIZE=2 \
DISC_EPOCHS=1 \
OUTPUT_DIR="$PHASE3_OUTPUT" \
./train.sh

echo ""
echo "Phase 3 complete. Output: $PHASE3_OUTPUT"
echo ""

# ============================================================
# 結果比較
# ============================================================

echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
echo ""

echo "Phase 1 (Vanilla Model):"
if [[ -f "$PHASE1_OUTPUT/training_history.json" ]]; then
    python3 -c "
import json
with open('$PHASE1_OUTPUT/training_history.json') as f:
    history = json.load(f)
    
print(f\"  Rounds: {len(history)}\")
if history:
    final = history[-1]
    print(f\"  Final Fool Rate: {final.get('fool_rate', 0)*100:.1f}%\")
    print(f\"  Final Mean P(real): {final.get('mean_fake_prob_true', 0):.3f}\")
"
fi

echo ""
echo "Phase 3 (LoRA Model):"
if [[ -f "$PHASE3_OUTPUT/training_history.json" ]]; then
    python3 -c "
import json
with open('$PHASE3_OUTPUT/training_history.json') as f:
    history = json.load(f)
    
print(f\"  Rounds: {len(history)}\")
if history:
    final = history[-1]
    print(f\"  Final Fool Rate: {final.get('fool_rate', 0)*100:.1f}%\")
    print(f\"  Final Mean P(real): {final.get('mean_fake_prob_true', 0):.3f}\")
"
fi

echo ""
echo "============================================================"
echo "All phases completed successfully!"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  Phase 1 (Vanilla): $PHASE1_OUTPUT"
echo "  LoRA Adapter:      $LORA_OUTPUT"
echo "  Phase 3 (LoRA):    $PHASE3_OUTPUT"
echo ""
echo "To use the trained LoRA in future runs:"
echo "  export GEN_MODE=lora"
echo "  export LORA_PATH=$LORA_OUTPUT"
echo "  ./train.sh"
echo ""
