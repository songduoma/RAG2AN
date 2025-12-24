# RAG²AN Runbook - 可重現執行指南

本文件提供從零開始執行 RAG²AN 專案的最小步驟，包含環境建立、資料準備、最小實驗執行與結果驗證。

## 前置需求檢查

在開始之前，確認以下項目：

- [ ] Python 3.10+ 已安裝
- [ ] CUDA 12.1 已安裝（本地生成器模式需要）
- [ ] 網路連線（下載 HuggingFace 資料集與模型）
- [ ] （可選）`OPENAI_API_KEY`（API 模式需要）

檢查 Python 版本：
```bash
python --version  # 應顯示 Python 3.10.x 或更高
```

檢查 CUDA（本地模式需要）：
```bash
nvidia-smi  # 應顯示 CUDA 版本與 GPU 資訊
```

## 步驟 1：建立環境

### 1.1 安裝 PyTorch（CUDA 12.1）

```bash
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

**注意：** 如果您的 CUDA 版本不同，請調整 PyTorch 版本。CPU 模式也可以執行，但本地生成器會較慢。

### 1.2 安裝專案依賴

```bash
# 假設您已在專案根目錄
pip install -r requirements.txt
```

### 1.3 安裝 FAISS

選擇 CPU 或 GPU 版本：

```bash
# CPU 版本（較慢但不需要 CUDA）
pip install faiss-cpu

# 或 GPU 版本（需要 CUDA）
pip install faiss-gpu
```

### 1.4 驗證安裝

```bash
python -c "import torch; import transformers; import datasets; print('✓ 核心套件已安裝')"
python -c "import faiss; print('✓ FAISS 已安裝')"
```

## 步驟 2：準備資料

### 2.1 資料自動下載

**不需要手動下載資料！** 資料集會在首次執行時自動從 HuggingFace 下載：

- **訓練資料：** `sanxing/advfake_news_please`（自動下載）
- **評估資料：** `sanxing/advfake`（自動下載或使用快取）

### 2.2 DPR 索引自動建立

DPR FAISS 索引會在首次呼叫檢索時自動建立：

- **位置：** `local/news-please/faiss_index/my_index.faiss`
- **時間：** 首次建立需要數分鐘（依資料集大小而定）
- **後續使用：** 自動載入已存在的索引

### 2.3 （可選）預先建立評估資料 Arrow 檔案

如果您想預先準備評估資料：

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('sanxing/advfake', split='train')
ds.with_format('arrow').save_to_disk('local/hf_datasets/advfake/advfake-train.arrow')
print('✓ 評估資料已準備')
"
```

## 步驟 3：執行最小實驗

### 3.1 方案 A：API 模式（推薦，不需要 GPU）

**優點：** 不需要 GPU，執行速度快，適合快速測試。

```bash
# 設定 API Key（必需）
export OPENAI_API_KEY="sk-your-api-key-here"

# 執行最小實驗
GEN_MODE=api \
OPENAI_MODEL=gpt-4o-mini \
DATASET_SPLIT=train[:64] \
NUM_ROUNDS=2 \
DISC_EPOCHS=1 \
BATCH_SIZE=2 \
GEN_SFT_EVERY_ROUND=0 \
OUTPUT_DIR=local/runs/smoke_test_api_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

**預期輸出：**
- 輸出目錄：`local/runs/smoke_test_api_YYYYMMDD_HHMMSS/`
- 包含：`round_1/`, `round_2/`, `disc_round_1/`, `disc_round_2/`, `disc_best/`, `training_history.json`

### 3.2 方案 B：本地模式（需要 GPU）

**優點：** 支援 LoRA SFT，生成器可持續改進。

```bash
# 執行最小實驗（本地模式）
GEN_MODE=local \
GEN_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
DATASET_SPLIT=train[:64] \
NUM_ROUNDS=2 \
DISC_EPOCHS=1 \
BATCH_SIZE=2 \
GEN_SFT_EVERY_ROUND=0 \
OUTPUT_DIR=local/runs/smoke_test_local_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

**注意：** 本地模式需要足夠的 GPU VRAM（建議 16GB+）。

### 3.3 方案 C：直接呼叫 Python 腳本

如果您想跳過 `train.sh` wrapper：

```bash
python -u scripts/gan_training.py \
  --dataset-name sanxing/advfake_news_please \
  --dataset-split train[:64] \
  --num-rounds 2 \
  --discriminator-epochs 1 \
  --batch-size 2 \
  --no-gen-sft \
  --output-dir local/runs/manual_test_$(date +%Y%m%d_%H%M%S) \
  --log-interval 10
```

## 步驟 4：驗證結果

### 4.1 檢查輸出目錄結構

```bash
# 替換為您的實際輸出目錄
OUTPUT_DIR="local/runs/smoke_test_api_20250101_120000"

# 檢查目錄是否存在
ls -la "$OUTPUT_DIR"

# 檢查必要檔案
ls "$OUTPUT_DIR/round_1/"          # 應包含生成的假樣本
ls "$OUTPUT_DIR/disc_round_1/"      # 應包含判別器 checkpoint
ls "$OUTPUT_DIR/disc_best/"         # 應包含最佳 checkpoint
cat "$OUTPUT_DIR/training_history.json"  # 應包含訓練歷史
```

### 4.2 檢查 training_history.json

```bash
python -c "
import json
with open('$OUTPUT_DIR/training_history.json') as f:
    history = json.load(f)
    print(f'總輪數: {len(history)}')
    for r in history:
        print(f\"Round {r['round']}: P(real)={r['mean_fake_prob_true']:.3f}, Fool Rate={r['fool_rate']:.1%}\")
"
```

### 4.3 檢查生成的樣本

```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('$OUTPUT_DIR/round_1')
print(f'生成的樣本數: {len(ds)}')
print('第一個樣本:')
print(ds[0])
"
```

## 步驟 5：執行評估（可選）

### 5.1 準備評估資料

如果尚未準備 Arrow 檔案：

```bash
python -c "
from datasets import load_dataset
from pathlib import Path
Path('local/hf_datasets/advfake').mkdir(parents=True, exist_ok=True)
ds = load_dataset('sanxing/advfake', split='train')
ds.with_format('arrow').save_to_disk('local/hf_datasets/advfake/advfake-train.arrow')
print('✓ 評估資料已準備')
"
```

### 5.2 評估判別器（無 RAG）

```bash
python scripts/evaluate_discriminator.py \
  --models-dir "$OUTPUT_DIR" \
  --dataset-path local/hf_datasets/advfake/advfake-train.arrow \
  --batch-size 8
```

### 5.3 評估判別器（含 RAG）

```bash
python scripts/evaluate_discriminator_with_rag.py \
  --models-dir "$OUTPUT_DIR" \
  --dataset-path local/hf_datasets/advfake/advfake-train.arrow \
  --rag-source dpr \
  --num-rag-results 3 \
  --batch-size 4
```

## Smoke Test（冒煙測試）

以下是最小的 smoke test 配置，用於快速驗證程式碼路徑是否正常。

### 最小配置

```bash
# 設定變數
export GEN_MODE=api  # 或 local（需要 GPU）
export OPENAI_API_KEY="sk-your-key"  # API 模式需要
export DATASET_SPLIT="train[:10]"    # 僅使用 10 個樣本
export NUM_ROUNDS=1                  # 僅執行 1 輪
export DISC_EPOCHS=1                 # 1 個 epoch
export BATCH_SIZE=2                  # 小 batch size
export GEN_SFT_EVERY_ROUND=0         # 停用 SFT
export OUTPUT_DIR="local/runs/smoke_$(date +%Y%m%d_%H%M%S)"

# 執行
./train.sh
```

### 驗證檢查清單

執行後，確認以下項目：

- [ ] **訓練完成無錯誤：** 終端機沒有 Python traceback
- [ ] **輸出目錄存在：** `$OUTPUT_DIR` 目錄已建立
- [ ] **round_1 存在：** `$OUTPUT_DIR/round_1/` 包含生成的樣本
- [ ] **checkpoint 存在：** `$OUTPUT_DIR/disc_round_1/` 包含模型檔案
- [ ] **歷史檔案存在：** `$OUTPUT_DIR/training_history.json` 存在且可讀取
- [ ] **歷史檔案格式正確：** JSON 包含至少一個 round 的資料

### 自動化 Smoke Test 腳本

建立 `scripts/run_smoke_test.sh`：

```bash
#!/bin/bash
set -euo pipefail

echo "=== RAG²AN Smoke Test ==="

# 設定最小配置
export GEN_MODE="${GEN_MODE:-api}"
export DATASET_SPLIT="train[:10]"
export NUM_ROUNDS=1
export DISC_EPOCHS=1
export BATCH_SIZE=2
export GEN_SFT_EVERY_ROUND=0
export OUTPUT_DIR="local/runs/smoke_$(date +%Y%m%d_%H%M%S)"

# API 模式檢查
if [[ "$GEN_MODE" == "api" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "❌ ERROR: OPENAI_API_KEY not set for API mode"
        exit 1
    fi
    echo "✓ API mode detected"
else
    echo "✓ Local mode detected (requires GPU)"
fi

# 執行訓練
echo "Running training..."
./train.sh

# 驗證結果
echo "Validating results..."
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "❌ ERROR: Output directory not created"
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR/round_1" ]]; then
    echo "❌ ERROR: round_1 directory not found"
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR/disc_round_1" ]]; then
    echo "❌ ERROR: disc_round_1 checkpoint not found"
    exit 1
fi

if [[ ! -f "$OUTPUT_DIR/training_history.json" ]]; then
    echo "❌ ERROR: training_history.json not found"
    exit 1
fi

# 檢查 JSON 格式
python -c "
import json
with open('$OUTPUT_DIR/training_history.json') as f:
    history = json.load(f)
    if len(history) == 0:
        raise ValueError('History is empty')
    print(f'✓ Found {len(history)} round(s) in history')
"

echo "✅ Smoke test passed!"
echo "Output directory: $OUTPUT_DIR"
```

使用方式：

```bash
chmod +x scripts/run_smoke_test.sh

# API 模式
GEN_MODE=api OPENAI_API_KEY="sk-xxx" ./scripts/run_smoke_test.sh

# 本地模式
GEN_MODE=local ./scripts/run_smoke_test.sh
```

## 常見問題與修補建議

### 問題 1：缺少 OPENAI_API_KEY

**症狀：**
```
⚠️  WARNING: OPENAI_API_KEY not set!
```

**解決方案：**
- **方案 A：** 設定 API key：`export OPENAI_API_KEY="sk-xxx"`
- **方案 B：** 使用本地模式：`GEN_MODE=local ./train.sh`

### 問題 2：CUDA 不可用（本地模式）

**症狀：**
```
RuntimeError: CUDA out of memory
或
AssertionError: CUDA not available
```

**解決方案：**
- **方案 A：** 使用 API 模式（不需要 GPU）
- **方案 B：** 檢查 CUDA 安裝：`nvidia-smi`
- **方案 C：** 降低 batch size：`BATCH_SIZE=1`

### 問題 3：資料集下載失敗

**症狀：**
```
OSError: Could not find dataset sanxing/advfake_news_please
```

**解決方案：**
- **方案 A：** 檢查網路連線
- **方案 B：** 設定 HuggingFace token：`export HF_TOKEN="hf_xxx"`
- **方案 C：** 使用本地資料集：`--dataset-path /path/to/local/dataset`

**TODO：** 如果 HuggingFace 資料集不可用，可以建立 toy data generator：

```python
# scripts/generate_toy_data.py
from datasets import Dataset
data = [{"description": f"Sample news article {i}." * 20} for i in range(100)]
ds = Dataset.from_list(data)
ds.save_to_disk("local/toy_data")
```

然後使用：`--dataset-path local/toy_data`

### 問題 4：FAISS 索引建立失敗

**症狀：**
```
ImportError: No module named 'faiss'
或
RuntimeError: FAISS index not found
```

**解決方案：**
```bash
pip install faiss-cpu  # 或 faiss-gpu
```

### 問題 5：記憶體不足（OOM）

**症狀：**
```
RuntimeError: CUDA out of memory
```

**解決方案：**
- 降低 batch size：`BATCH_SIZE=1`
- 降低序列長度：`MAX_LENGTH=256`
- 使用較小的模型：`DISC_MODEL=microsoft/deberta-v3-small`
- 停用 RAG：`GEN_USE_RAG=0 DISC_USE_RAG=0`

### 問題 6：DPR 索引建立時間過長

**症狀：**
首次執行時卡在 "building DPR context embeddings..."

**解決方案：**
- 這是正常行為，首次建立需要數分鐘
- 索引會快取在 `local/news-please/faiss_index/my_index.faiss`
- 後續執行會自動載入，無需重新建立

### 問題 7：評估資料 Arrow 檔案不存在

**症狀：**
```
FileNotFoundError: Dataset not found: local/hf_datasets/advfake/advfake-train.arrow
```

**解決方案：**
```bash
python -c "
from datasets import load_dataset
from pathlib import Path
Path('local/hf_datasets/advfake').mkdir(parents=True, exist_ok=True)
ds = load_dataset('sanxing/advfake', split='train')
ds.with_format('arrow').save_to_disk('local/hf_datasets/advfake/advfake-train.arrow')
"
```

## 完整範例：從零到結果

以下是一個完整的範例，從環境建立到產生結果：

```bash
# 1. 進入專案目錄（替換為您的實際路徑）
cd /path/to/RAG2AN

# 2. 建立虛擬環境（可選但推薦）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 3. 安裝依賴
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install faiss-cpu  # 或 faiss-gpu

# 4. 設定 API key（API 模式）
export OPENAI_API_KEY="sk-your-api-key"

# 5. 執行 smoke test
GEN_MODE=api \
DATASET_SPLIT=train[:10] \
NUM_ROUNDS=1 \
DISC_EPOCHS=1 \
BATCH_SIZE=2 \
GEN_SFT_EVERY_ROUND=0 \
OUTPUT_DIR=local/runs/first_run_$(date +%Y%m%d_%H%M%S) \
./train.sh

# 6. 檢查結果
OUTPUT_DIR=$(ls -td local/runs/*/ | head -1)
echo "Output directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
cat "$OUTPUT_DIR/training_history.json"
```

## 下一步

完成 smoke test 後，您可以：

1. **增加資料量：** `DATASET_SPLIT=train[:1000]`
2. **增加輪數：** `NUM_ROUNDS=10`
3. **啟用 LoRA SFT：** `GEN_SFT_EVERY_ROUND=1`（本地模式）
4. **啟用 RAG：** `GEN_USE_RAG=1 DISC_USE_RAG=1`
5. **調整超參數：** 見 `train.sh` 中的環境變數

更多詳細資訊請參考 [REPO_MAP.md](REPO_MAP.md)。

