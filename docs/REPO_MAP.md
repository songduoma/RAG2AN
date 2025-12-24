# RAG²AN Repository Map

## 目錄結構與用途

```
RAG2AN/
├── src/                          # 核心模組
│   ├── __init__.py
│   ├── generator.py              # 假新聞生成器（支援 API/本地模式）
│   ├── discriminator.py          # DeBERTa 判別器（提供 VAF 回饋）
│   ├── retrieval_dpr.py         # DPR 檢索模組（封裝 get_dpr）
│   └── news.py                   # DPR 實作與 FAISS 索引管理
│
├── scripts/                      # 訓練與評估腳本
│   ├── gan_training.py           # 主訓練入口（GAN 循環）
│   ├── evaluate_discriminator.py # 判別器評估（無 RAG）
│   ├── evaluate_discriminator_with_rag.py  # 判別器評估（含 RAG）
│   ├── evaluate_discriminator_baseline.py   # Baseline 評估
│   ├── evaluate_discriminator_with_rag_baseline.py  # Baseline RAG 評估
│   ├── inspect_arrow.py          # Arrow 檔案檢查工具
│   ├── show_vattention.py        # 注意力視覺化
│   └── search.py                 # 搜尋工具
│
├── train.sh                      # 訓練腳本 wrapper（設定環境變數）
├── requirements.txt              # Python 依賴套件
├── README.md                     # 專案說明
│
└── local/                        # 執行時產生（不在 repo 中）
    ├── rag_gan_runs/            # 訓練輸出目錄
    │   └── <timestamp>/         # 每次執行的輸出
    │       ├── round_1/         # 第 1 輪生成的假樣本（HF dataset）
    │       ├── disc_round_1/    # 第 1 輪判別器 checkpoint
    │       ├── disc_best/       # 最佳判別器 checkpoint
    │       ├── training_history.json  # 訓練歷史摘要
    │       └── train.log        # 訓練日誌（如果設定 LOG_FILE）
    │
    ├── news-please/             # DPR 索引快取
    │   └── faiss_index/
    │       └── my_index.faiss  # FAISS 索引檔案（首次執行時建立）
    │
    ├── hf_datasets/             # HuggingFace 資料集快取（可選）
    │   └── advfake/
    │       └── advfake-train.arrow
    │
    └── rag_cache/               # RAG 評估快取（可選）
```

## 單一真入口（Entry Points）

### 1. 訓練（Training）

**主要入口：** `train.sh` → `scripts/gan_training.py`

```bash
# 使用 train.sh（推薦）
./train.sh

# 或直接呼叫 Python 腳本
python -u scripts/gan_training.py \
  --dataset-name sanxing/advfake_news_please \
  --dataset-split train[:1000] \
  --num-rounds 2 \
  --output-dir local/runs/manual_call
```

**讀取的配置：**
- 環境變數：`GEN_MODE`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `GEN_MODEL`, `GEN_USE_LORA`, `GEN_SFT_EVERY_ROUND`, `DISC_MODEL`, `RAG_SOURCE`, `DISC_USE_RAG`, `GEN_USE_RAG`, `USE_VAF_FEEDBACK`, `USE_VAF_FEWSHOT` 等（見 `train.sh`）
- CLI 參數：見下方「CLI 參數清單」

### 2. 評估（Evaluation）

**判別器評估（無 RAG）：** `scripts/evaluate_discriminator.py`

```bash
python scripts/evaluate_discriminator.py \
  --models-dir local/rag_gan_runs/<run_name> \
  --dataset-path local/hf_datasets/advfake/advfake-train.arrow \
  --batch-size 8
```

**判別器評估（含 RAG）：** `scripts/evaluate_discriminator_with_rag.py`

```bash
python scripts/evaluate_discriminator_with_rag.py \
  --models-dir local/rag_gan_runs/<run_name> \
  --dataset-path local/hf_datasets/advfake/advfake-train.arrow \
  --rag-source dpr \
  --num-rag-results 3 \
  --batch-size 4
```

**讀取的配置：**
- CLI 參數：`--models-dir`, `--dataset-path`, `--batch-size`, `--rag-source`, `--num-rag-results`, `--disc-use-rag`, `--bf16`, `--max-length`
- 環境變數：`RAG_SOURCE`, `NUM_RAG_RESULTS`, `DISC_USE_RAG`（作為預設值）

### 3. 工具腳本

- `src/generator.py` - 獨立生成器示範（`if __name__ == "__main__"`）
- `src/news.py` - DPR 互動式搜尋（`if __name__ == "__main__"`）
- `scripts/inspect_arrow.py` - Arrow 檔案檢查
- `scripts/show_vattention.py` - 注意力視覺化
- `scripts/search.py` - 搜尋工具

## 環境需求

### Python 版本
- **Python 3.10+**（必需）

### CUDA 與 PyTorch
- **CUDA 12.1**（本地生成器模式需要 GPU）
- **PyTorch 2.6.0+cu118**（已固定在 `requirements.txt`）
  - 安裝方式：`pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118`

### 關鍵套件（requirements.txt）
```
accelerate>=0.27.0
datasets>=2.16.0
google-generativeai>=0.7.0
google-search-results>=2.4.2
openai>=1.0.0
requests>=2.31.
safetensors>=0.4.3
scikit-learn>=1.3.0
sentencepiece>=0.1.99
transformers>=4.46.0
peft>=0.13.0
```

**注意：** 需要 FAISS 支援（CPU 或 GPU）
- CPU: `pip install faiss-cpu`
- GPU: `pip install faiss-gpu`（需要 CUDA）

### 環境變數（可選）
- `OPENAI_API_KEY` - API 模式生成器需要（格式：`sk-xxx`）
- `HF_TOKEN` - 存取 gated HuggingFace 模型需要
- `GEN_MODE` - 生成器模式：`api` 或 `local`（預設：`local`）
- `OPENAI_MODEL` - API 模式使用的模型（預設：`gpt-4o-mini`）
- `OPENAI_BASE_URL` - API 基礎 URL（預設：`https://api.openai.com/v1`）

### 安裝指令

```bash
# 1. 安裝 PyTorch（CUDA 12.1）
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# 2. 安裝其他依賴
pip install -r requirements.txt

# 3. 安裝 FAISS（選擇 CPU 或 GPU）
pip install faiss-cpu  # 或 faiss-gpu
```

## 資料來源

### 訓練資料集
- **來源：** `sanxing/advfake_news_please`（HuggingFace）
- **下載方式：** 自動下載（首次執行時）
- **格式：** HuggingFace Dataset，欄位包含 `description`（或 `text`）
- **前處理：**
  - 移除長度 < 50 字元的描述（過濾短樣本）
  - 排除 `sanxing/advfake` 中的 402 個真實描述（避免資料洩漏）
  - 支援資料集切片（如 `train[:10000]`）

### 評估資料集
- **來源：** `sanxing/advfake`（HuggingFace）
- **用途：** 評估判別器效能
- **格式：** 402 個真實/假新聞配對
  - `description` - 真實描述
  - `f_description` - 假新聞重寫
- **下載方式：** 自動下載或使用快取 Arrow 檔案
  - 預設路徑：`local/hf_datasets/advfake/advfake-train.arrow`
  - 或從 HuggingFace cache：`~/.cache/huggingface/datasets/sanxing___advfake/...`

### DPR 檢索索引
- **來源：** `sanxing/advfake_news_please`（與訓練資料相同）
- **建立方式：** 首次呼叫 `get_dpr()` 時自動建立
- **儲存位置：** `local/news-please/faiss_index/my_index.faiss`
- **模型：**
  - Context Encoder: `facebook/dpr-ctx_encoder-single-nq-base`
  - Question Encoder: `facebook/dpr-question_encoder-single-nq-base`
- **處理時間：** 首次建立需要數分鐘（依資料集大小而定）

### 資料快取位置
- HuggingFace 資料集：`~/.cache/huggingface/datasets/`
- DPR FAISS 索引：`local/news-please/faiss_index/`
- RAG 評估快取：`local/rag_cache/`（可選）

## 輸出與 Checkpoint 存放規則

### 輸出目錄結構

當設定 `OUTPUT_DIR`（預設：`local/rag_gan_runs/<timestamp>`）時，會產生以下結構：

```
OUTPUT_DIR/
├── training_history.json        # 每輪訓練摘要（JSON）
├── train.log                    # 訓練日誌（如果設定 LOG_FILE）
├── round_1/                     # 第 1 輪生成的假樣本
│   ├── dataset_info.json        # HF dataset metadata
│   ├── state.json
│   └── *.arrow                  # 生成的假新聞樣本
├── round_2/                     # 第 2 輪生成的假樣本
│   └── ...
├── disc_round_1/                # 第 1 輪判別器 checkpoint
│   ├── config.json
│   ├── pytorch_model.bin        # 或 model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── ...
├── disc_round_2/                # 第 2 輪判別器 checkpoint
│   └── ...
└── disc_best/                   # 最佳判別器 checkpoint（最低 loss）
    └── ...
```

### Checkpoint 命名規則

- **每輪判別器：** `disc_round_{round_id}/`（每輪結束時儲存）
- **最佳判別器：** `disc_best/`（當 loss 低於歷史最佳時更新）
- **生成樣本：** `round_{round_id}/`（HF dataset 格式）

### training_history.json 格式

每輪記錄包含：
```json
{
  "round": 1,
  "mean_fake_prob_true": 0.234,
  "fool_rate": 0.15,
  "avg_disc_loss": 0.6234,
  "disc_checkpoint": "local/rag_gan_runs/.../disc_round_1",
  "disc_best_checkpoint": "local/rag_gan_runs/.../disc_best",
  "num_generated": 100,
  "num_real_samples": 500,
  "skipped_disc_training": false,
  "detection_reasons": {...},
  "confidence_distribution": {...}
}
```

### 日誌輸出

- **標準輸出：** 訓練進度、每輪摘要、最終統計
- **LOG_FILE：** 如果設定 `LOG_FILE`，會 tee 到檔案（預設：`$OUTPUT_DIR/train.log`）

## CLI 參數清單

### scripts/gan_training.py

#### 資料相關
- `--dataset-name` (str, default: `sanxing/advfake_news_please`) - HuggingFace 資料集名稱
- `--dataset-split` (str, default: `train`) - 資料集分割（支援切片如 `train[:1000]`）
- `--dataset-path` (str, optional) - 本地資料集路徑（使用 `load_from_disk`）

#### 訓練循環
- `--num-rounds` (int, default: 2) - GAN 訓練輪數
- `--discriminator-epochs` (int, default: 1) - 每輪判別器訓練 epoch 數
- `--batch-size` (int, default: 1) - 判別器 batch size
- `--lr` (float, default: 3e-5) - 判別器學習率
- `--max-length` (int, default: 512) - 最大序列長度
- `--seed` (int, default: 42) - 隨機種子

#### RAG 設定
- `--rag-source` (str, choices: `dpr|none`, default: `dpr`) - 檢索來源
- `--gen-use-rag` / `--no-gen-rag` - 生成器是否使用 RAG
- `--disc-use-rag` / `--no-disc-rag` - 判別器是否使用 RAG
- `--num-rag-results` (int, default: 3) - RAG 檢索結果數量

#### 模型設定
- `--generator-model` (str, default: `Qwen/Qwen3-4B-Instruct-2507`) - 生成器模型 ID
- `--discriminator-model` (str, default: `microsoft/deberta-v3-base`) - 判別器模型 ID
- `--disc-positive-label-id` (int, optional) - 覆蓋判別器正標籤 ID

#### 生成器 LoRA SFT
- `--gen-sft-every-round` / `--no-gen-sft` - 每輪是否執行生成器 SFT
- `--gen-sft-lr` (float, default: 5e-5) - SFT 學習率
- `--gen-sft-steps` (int, default: 2) - SFT 梯度步數
- `--gen-sft-batch-size` (int, default: 1) - SFT batch size
- `--gen-sft-max-length` (int, default: 512) - SFT 最大序列長度
- `--gen-sft-success-threshold` (float, default: 0.55) - 成功樣本閾值（P(real)）
- `--gen-sft-max-samples` (int, default: 2) - SFT 使用的最大樣本數
- `--gen-sft-lambda-kl` (float, default: 0.01) - KL 正則化權重
- `--gen-sft-max-grad-norm` (float, default: 1.0) - 梯度裁剪
- `--gen-sft-warmup-rounds` (int, default: 3) - 暖身輪數

#### 樣本採樣
- `--real-samples-per-round` (int, optional) - 每輪使用的真實樣本數
- `--gen-real-fixed` / `--no-gen-real-fixed` - 生成器真實樣本是否固定
- `--disc-real-samples-per-round` (int, optional) - 判別器每輪真實樣本數
- `--disc-real-sampling` (str, choices: `random|cycle`, default: `random`) - 真實樣本採樣方式
- `--disc-real-exclude-gen` / `--no-disc-real-exclude-gen` - 是否排除生成器使用的真實樣本

#### 動態平衡
- `--label-smoothing` (float, default: 0.1) - 標籤平滑係數
- `--min-fool-rate-to-train` (float, default: 0.05) - 最低 fool rate 才訓練判別器
- `--max-skip-rounds` (int, default: 2) - 最多連續跳過判別器訓練輪數

#### VAF（Verbal Adversarial Feedback）
- `--use-vaf-feedback` / `--no-vaf-feedback` - 啟用判別器回饋區塊
- `--use-vaf-fewshot` / `--no-vaf-fewshot` - 啟用成功範例 few-shot

#### 輸出與日誌
- `--output-dir` (str, optional) - 輸出目錄
- `--log-interval` (int, default: 10) - 進度日誌間隔

### scripts/evaluate_discriminator.py

- `--model-path` (str, optional) - 單一判別器 checkpoint 路徑
- `--models-dir` (Path, default: `local/rag_gan_runs/20251223_132301`) - 包含 `disc_round_*` 的目錄
- `--dataset-path` (Path, default: `local/hf_datasets/advfake/advfake-train.arrow`) - advfake Arrow 檔案路徑
- `--batch-size` (int, default: 8) - 推論 batch size
- `--disc-use-rag` / `--no-disc-rag` - 是否使用 RAG
- `--rag-source` (str, choices: `dpr|none`, default: `dpr`) - 檢索來源
- `--num-rag-results` (int, default: 3) - RAG 結果數量
- `--bf16` (flag, default: True) - 使用 bfloat16
- `--max-length` (int, default: 512) - 最大序列長度

### scripts/evaluate_discriminator_with_rag.py

- `--models-dir` (Path, default: `local/rag_gan_runs/G-D+`) - 包含 `disc_round_*` 的目錄
- `--dataset-path` (Path, default: `local/hf_datasets/advfake/advfake-train.arrow`) - advfake Arrow 檔案路徑
- `--rag-source` (str, choices: `dpr|none`, default: `dpr`) - 檢索來源
- `--num-rag-results` (int, default: 3) - RAG 結果數量
- `--rag-cache-path` (Path, optional) - RAG 快取路徑
- `--batch-size` (int, default: 4) - 推論 batch size
- `--bf16` (flag, default: True) - 使用 bfloat16
- `--max-length` (int, default: 512) - 最大序列長度

