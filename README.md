# RAG2AN – Retrieval-Augmented Generator & Analyzer for Fake News GAN

> 研究用途限定：本程式碼僅用於假新聞偵測與對抗式生成的研究。請勿用於散播錯誤資訊。

RAG2AN 是一個「生成器 (Generator) + 判別器 (Discriminator)」的簡易 GAN 迴圈，用檢索增強生成 (RAG) 來生成可疑新聞，再透過編碼式模型判別真偽並回饋可疑詞，持續迭代提升判別能力。

## 專案結構
- `gan_training.py`：主訓練迴圈，負責載入資料、呼叫生成器、訓練判別器、保存檢查點。
- `train.sh`：方便的啟動腳本，透過環境變數快速調整資料來源、回合數與模型。
- `generator.py`：以 `Qwen/Qwen2.5-7B-Instruct` 為預設生成器，內建 Wikipedia RAG，支援 feedback prompt。
- `discriminator.py`：以 DeBERTa 為編碼式判別器，可回傳可疑詞 (attention-based)。
- `search.py`：Google Search (SerpAPI) RAG 支援。
- `inspect_gan_run.py`：檢視輸出資料夾，快速瀏覽各回合生成結果與統計。

## 環境需求與安裝
- Python 3.10、具備 CUDA 12.1 GPU（Qwen2.5-7B-Instruct 需 16GB+ 顯存；資源不足可改小模型，例如 `gpt2`）。
- HF Token：使用 Qwen 7B 需先在 Hugging Face 同意條款並設定 `HF_TOKEN`。
- Google RAG：需 `SERPAPI_API_KEY`（安裝時已含 `google-search-results`）。
- OpenAI / Gemini：僅在 `utils.py` 的相關函式被呼叫時才需 `OPENAI_API_KEY` / `GEMINI_API_KEY`。

安裝步驟：
```bash
cd RAG2AN
# 若要自建環境，請先建立/啟動你的 venv 或 conda env
pip install -r requirements.txt   # 內含 cu121 的 PyTorch extra-index
```
> 如果你的機器不是 CUDA 12.1，請將 `requirements.txt` 開頭的 `--extra-index-url` 改成對應版本，或自行安裝相符的 torch/torchvision/torchaudio。

## 快速開始
使用預設參數（CNN/DailyMail 取少量樣本，Wiki RAG，10 回合 GAN）：
```bash
cd RAG2AN
./train.sh
```

調整環境變數即可客製化（以下示範縮短回合、改成本地輸出路徑並啟用判別器 RAG）：
```bash
DATASET_NAME=cnn_dailymail \
DATASET_CONFIG=3.0.0 \
DATASET_SPLIT=train[:64] \
NUM_ROUNDS=3 \
DISC_EPOCHS=1 \
BATCH_SIZE=4 \
RAG_SOURCE=wiki \      # 選項：wiki | google | none
DISC_USE_RAG=1 \
GEN_USE_WIKI=1 \
OUTPUT_DIR=local/runs/demo_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

`train.sh` 會呼叫 `gan_training.py`，主要參數說明：
- `--dataset-name` / `--dataset-config` / `--dataset-split`：HF 資料集設定，預設 `sanxing/advfake_news_please` (在 `gan_training.py`)；`train.sh` 預設改成 `cnn_dailymail 3.0.0 train`。
- `--max-samples`：限制樣本數便於試跑。
- `--num-rounds`：GAN 迭代回合數（每回合：生成假新聞 + 訓練判別器）。
- `--rag-source`：`wiki` | `google` | `none`，決定生成器/判別器取得的上下文來源。
- `--disc-use-rag` / `--generator-use-wiki` / `--filter-no-wiki`：是否給判別器 RAG 上下文、生成器是否允許 wiki fallback、是否過濾查無 wiki 的樣本。
- `--generator-model` / `--discriminator-model`：HF 模型名稱。
- `--output-dir`：每回合的生成樣本與判別器 checkpoint 會寫入此路徑。

## 產出內容
指定 `--output-dir` 時會得到：
```
output_dir/
  round_1/                # 以 HF Dataset 儲存的生成結果 (text, label, prob_true, suspicious_words, feedback_prompt)
  round_2/
  ...
  disc_round_1/           # 每回合的判別器 checkpoint (transformers 標準格式)
  disc_best/              # 目前最佳 loss 的判別器
```
可用 `inspect_gan_run.py` 快速檢查（以互動方式指定路徑）：
```bash
python - <<'PY'
from inspect_gan_run import inspect_run
inspect_run("local/runs/demo_20250101_120000", sample_n=3)
PY
```
（或編輯 `inspect_gan_run.py` 內的 `__main__`，將 `run_dir` 改成你的輸出資料夾。）

## 流程概覽
1) 載入資料 (`load_dataset` 或 `load_from_disk`)，必要欄位：`title` / `description` 或 `text`。CNN/DailyMail 會自動萃取 `title` 及日期。
2) 依 `rag_source` 取得上下文：Wikipedia API（預設）、或 SerpAPI Google、或關閉 RAG。
3) 生成器產生假新聞，僅留下 `Title/Body` 供判別器使用，避免把 prompt 餵給模型。
4) 判別器回傳真實機率與可疑詞，feedback 會寫回樣本，供下一回合生成器調整。
5) 混合真/假樣本後訓練判別器，保存回合統計與 checkpoint。

## 注意事項
- 記憶體與顯示卡需求取決於生成器模型大小；若資源有限，可替換 `generator_model` 為較小的 HF 模型（例如 `gpt2`）以測試流程。
- 預設 requirements 使用 PyTorch 2.5.1 + cu121 並搭配 `safetensors`，避免舊版 torch 的 `torch.load` 安全限制。如需其他 CUDA 版本請調整對應輪檔。
- Wikipedia API 需可連線；若被擋或逾時，`generator.py` 會給 fallback 文字，確保流程不中斷。
- 使用 Google RAG 時請遵守 SerpAPI 條款；請勿對真實搜尋結果進行未授權的資訊蒐集。
- 本程式碼僅供研究與防禦用途，不得用於實際散播假訊息。
