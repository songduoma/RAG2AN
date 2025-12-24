# Research Reframe: From GAN to Detection

本文檔記錄 RAG²AN 專案從 **Fake News GAN** 到 **Fake News Detection** 的研究重新定位，包含現有架構分析、Detection 主軸敘事、程式碼對應與新實驗矩陣。

---

## 第一部分：現有研究架構分析

### 研究問題（Research Problem）

現有專案聚焦於 **生成對抗網路（GAN）框架下的假新聞生成與判別**。研究問題可表述為：如何在 GAN 訓練循環中，讓生成器（Generator）產生更逼真的假新聞，同時讓判別器（Discriminator）學會區分真實與生成的樣本。核心挑戰在於平衡生成器與判別器的對抗動態，避免判別器過早壓制生成器，導致生成器無法學習改進。

專案採用 **Verbal Adversarial Feedback (VAF)** 機制，讓判別器不僅輸出二元分類結果，還提供自然語言形式的回饋（包括可疑詞彙、檢測原因、改進建議），生成器根據這些回饋調整生成策略。此外，系統整合了 RAG（Retrieval-Augmented Generation）機制，透過 DPR 檢索真實新聞作為風格參考，提升生成內容的逼真度。

### 方法論（Methodology）

**GAN 訓練循環：** 系統執行多輪對抗訓練，每輪包含以下步驟：(1) 生成器基於真實新聞樣本生成假新聞；(2) 判別器對生成的假新聞進行評估，輸出 P(real) 機率、可疑詞彙、檢測原因與改進建議；(3) 根據 VAF 機制，將判別器的自然語言回饋整合到下一輪生成器的 prompt 中；(4) 判別器在混合的真實與生成樣本上進行監督學習，更新參數。

**VAF 回饋機制：** VAF 包含兩個核心元件：(a) **Feedback Block**：判別器提供結構化的自然語言回饋，包括檢測結果（REAL/FAKE）、信心水準、識別的問題（如 sensationalist_language、vague_attribution、factual_inconsistency、style_mismatch）、標記的可疑詞彙，以及具體的改進建議；(b) **Few-shot Cache**：系統維護一個成功案例快取（最多 5 個成功騙過判別器的假新聞樣本），在後續生成時作為 few-shot 範例注入 prompt，引導生成器模仿成功策略。

**動態平衡機制：** 為避免判別器過早壓制生成器，系統實作了動態平衡策略：(1) **Label Smoothing**：軟化判別器的學習信號；(2) **Fool-rate Triggered Pauses**：當生成器的 fool rate（成功騙過判別器的比例）低於閾值時，暫停判別器訓練，給予生成器改進時間；(3) **Forced Resumes**：設定最大跳過輪數，確保判別器不會長期停滯。

**LoRA Mini-SFT：** 在本地模式（非 API 模式）下，系統支援對生成器進行 LoRA 微調。每輪訓練結束後，選取成功騙過判別器且 P(real) 超過閾值的樣本，對生成器進行少量梯度步數的 KL-正則化微調，使生成器參數逐步適應判別器的檢測模式。

### 實驗設計（Experiments）

**訓練流程：** 實驗採用多輪 GAN 訓練循環，預設執行 6-10 輪。每輪使用 `sanxing/advfake_news_please` 資料集的真實新聞樣本作為生成器輸入，生成對應的假新聞。判別器在每輪混合真實與生成的樣本上進行訓練（預設 1 epoch），並保存每輪的 checkpoint。

**資料處理：** 系統過濾長度 < 50 字元的短樣本，並排除 `sanxing/advfake` 資料集中的 402 個真實描述，避免評估時的資料洩漏。支援資料集切片（如 `train[:10000]`）以控制訓練規模。

**評估方式：** 訓練結束後，使用 `scripts/evaluate_discriminator.py` 和 `scripts/evaluate_discriminator_with_rag.py` 在 `sanxing/advfake` 資料集（402 個真實/假新聞配對）上評估判別器效能。評估支援兩種模式：(1) 無 RAG：直接使用原始文本；(2) 含 RAG：使用與訓練時相同的 DPR 檢索上下文格式。

### 評估指標（Metrics）

**生成器指標：**
- **Fool Rate**：成功騙過判別器的假新聞比例（P(real) > 0.5 視為成功）
- **Mean P(real)**：生成樣本的平均 P(real) 機率（越高表示生成器越成功）
- **P(real) Range**：最小與最大 P(real) 值，反映生成樣本的品質分布

**判別器指標：**
- **Average Discriminator Loss**：判別器在混合樣本上的平均損失（越低表示判別能力越強）
- **Macro-F1**：在評估資料集上的宏觀 F1 分數
- **ROC-AUC**：ROC 曲線下面積，衡量判別器的整體分類能力
- **Accuracy**：整體分類準確率

**檢測分析指標：**
- **Confidence Distribution**：高/中/低信心樣本的分布（基於 P(fake) 閾值：≥0.8 高、≥0.6 中、否則低）
- **Detection Reasons Distribution**：各類檢測原因（sensationalist_language、vague_attribution、factual_inconsistency、style_mismatch）的出現頻率

**訓練動態指標：**
- **Discriminator Training Status**：每輪是否跳過判別器訓練（基於 fool rate 閾值）
- **Successful Examples Count**：few-shot cache 中成功案例的數量

---

## 第二部分：Detection 主軸敘事（Reframed Narrative）

### 研究問題（Problem）

**假新聞檢測的核心挑戰：** 現有的假新聞檢測系統面臨三大挑戰：(1) **對抗魯棒性不足**：檢測器容易受到精心設計的對抗樣本攻擊，攻擊者可以透過微調語言風格、替換可疑詞彙等方式規避檢測；(2) **語意對齊困難**：檢測器需要理解真實新聞的語意特徵與寫作模式，但傳統的監督學習方法往往依賴表面特徵（如特定詞彙、句式），缺乏深層語意理解；(3) **樣本多樣性不足**：訓練資料中的假新聞樣本往往缺乏多樣性，導致檢測器對新型攻擊模式泛化能力弱。

**VAF 如何解決這些挑戰：** Verbal Adversarial Feedback (VAF) 機制透過自然語言回饋，讓檢測器不僅輸出分類結果，還能解釋檢測原因並提供改進建議。這種機制在 GAN 訓練循環中產生更難的對抗樣本，同時提升檢測器的語意對齊能力與對抗魯棒性。

### 方法論（Method）

**VAF 增強檢測的三個機制：**

**(1) 語意對齊提升（Better Semantic Alignment）**

VAF 透過自然語言回饋，讓生成器學習真實新聞的深層語意特徵，而非僅模仿表面模式。判別器提供的改進建議（如「使用具體、具名的來源而非模糊短語」）引導生成器理解真實新聞的寫作規範，生成的樣本更接近真實新聞的語意空間。當這些樣本用於訓練判別器時，判別器被迫學習更深層的語意特徵，而非僅依賴關鍵詞匹配。

**程式碼對應：** `src/discriminator.py` 的 `get_detailed_feedback()` 方法（第 266-453 行）分析文本的語意特徵，識別 sensationalist_language、vague_attribution、factual_inconsistency、style_mismatch 等檢測原因，並生成對應的改進建議。這些建議透過 `scripts/gan_training.py` 的 VAF 回饋建構邏輯（第 556-606 行）整合到生成器 prompt 中。

**(2) 生成更難樣本（Harder Adversarial Samples）**

VAF 的迭代回饋機制讓生成器逐步改進，產生越來越難檢測的假新聞。每輪訓練中，判別器識別生成樣本的問題並提供改進建議，生成器根據建議調整策略，產生更逼真的樣本。這種對抗性訓練產生了一個「難樣本生成器」，持續產生挑戰檢測器極限的樣本。

**程式碼對應：** `scripts/gan_training.py` 的 `run_round()` 方法（第 442-819 行）執行每輪訓練循環。第 545-551 行呼叫 `get_detailed_feedback()` 獲取判別器回饋，第 558-606 行建構 VAF 回饋 prompt，第 500-507 行將回饋注入生成器。第 630-633 行計算 fool rate，追蹤生成器的改進進度。

**(3) 對抗魯棒性提升（Adversarial Robustness）**

VAF 訓練循環讓檢測器持續面對改進中的對抗樣本，迫使檢測器學習更魯棒的特徵表示。當生成器根據回饋調整策略時，檢測器必須適應新的攻擊模式，學習更通用的檢測規則。這種動態對抗過程提升了檢測器對未知攻擊模式的泛化能力。

**程式碼對應：** `scripts/gan_training.py` 的動態平衡機制（第 677-687 行）根據 fool rate 決定是否訓練判別器。當 fool rate 低於閾值時暫停訓練，給予生成器改進時間；當生成器改進後，判別器重新訓練以適應新的攻擊模式。這種機制確保檢測器持續面對挑戰，而非過早壓制生成器。

### 實驗設計（Experiments）

**Baseline Detection：** 首先訓練一個標準的判別器（無 VAF），在真實與初始生成的假新聞樣本上進行監督學習。此 baseline 使用與 VAF 版本相同的架構（DeBERTa-base）與訓練設定，但生成器不接收判別器回饋，僅進行標準的假新聞生成。

**VAF 增強檢測：** 在 GAN 訓練循環中啟用 VAF 機制，讓判別器提供自然語言回饋，生成器根據回饋改進。訓練多輪後，比較 VAF 訓練的判別器與 baseline 在評估資料集上的效能差異。

**Ablation Studies：**

- **VAF Feedback Only**：僅啟用 feedback block，停用 few-shot cache，評估自然語言回饋單獨的貢獻。
- **VAF Few-shot Only**：僅啟用 few-shot cache，停用 feedback block，評估成功案例示範的貢獻。
- **Feedback Intensity Variations**：調整 VAF 回饋的強度（如改變檢測原因數量、改進建議詳細程度），評估不同回饋強度對檢測效能的影響。

**對抗魯棒性評估：** 在評估階段，使用 VAF 訓練循環中生成的假新聞樣本作為對抗測試集，比較 baseline 與 VAF 增強檢測器在這些樣本上的表現，量化對抗魯棒性的提升。

### 評估指標（Metrics）

**檢測效能指標：**
- **Macro-F1**：在 `sanxing/advfake` 評估資料集上的宏觀 F1 分數（主要指標）
- **ROC-AUC**：ROC 曲線下面積，衡量檢測器的整體分類能力
- **Accuracy**：整體分類準確率

**對抗魯棒性指標：**
- **Adversarial Fool Rate**：在 VAF 訓練循環生成的對抗樣本上，檢測器被騙過的比例（越低表示魯棒性越強）
- **Adversarial Detection Rate**：在對抗樣本上的正確檢測率（1 - Fool Rate）

**語意對齊指標：**
- **Detection Reasons Diversity**：檢測原因的多樣性（不同原因類型的分布），反映檢測器對不同語意特徵的敏感度
- **Confidence Calibration**：檢測器信心水準與實際準確率的對齊程度（高信心樣本應有更高的準確率）

**訓練動態指標：**
- **Fool Rate Progression**：各輪訓練中生成器的 fool rate 變化趨勢，反映對抗樣本難度的提升
- **Discriminator Loss Progression**：各輪訓練中判別器損失的變化趨勢，反映檢測能力的改進

---

## 第三部分：程式碼對應（Code Mapping）

### Detection 模型（Detection Models）

**核心類別：** `src/discriminator.py` 中的 `EncoderDiscriminator` 類別

- **類別定義：** 第 78-145 行
- **初始化：** `__init__()` 方法（第 78-145 行），載入預訓練的 DeBERTa 模型（預設 `microsoft/deberta-v3-base`）
- **前向傳播：** `forward()` 方法（繼承自 `AutoModelForSequenceClassification`）
- **機率預測：** `predict()` 方法（第 146-158 行），輸出 P(real) 機率
- **可疑詞彙識別：** `get_suspicious_words()` 方法（第 199-264 行），使用注意力機制識別可疑詞彙
- **詳細回饋生成：** `get_detailed_feedback()` 方法（第 266-453 行），生成包含檢測原因、改進建議的結構化回饋

**輔助函數：**
- `get_encoder_discriminator()`：第 455-470 行，建立 EncoderDiscriminator 實例的工廠函數
- `format_discriminator_input()`：第 512-544 行，格式化判別器輸入（可選 RAG 上下文）

### GAN 訓練循環（GAN Training Loop）

**核心類別：** `scripts/gan_training.py` 中的 `GANTrainer` 類別

- **類別定義：** 第 98-441 行
- **初始化：** `__init__()` 方法（第 110-213 行），初始化生成器、判別器、優化器、VAF 設定
- **單輪訓練：** `run_round()` 方法（第 442-819 行），執行一輪完整的 GAN 訓練循環
  - 第 472-529 行：生成器生成假新聞
  - 第 531-551 行：判別器評估生成的假新聞並生成回饋
  - 第 556-606 行：建構 VAF 回饋 prompt（feedback block + few-shot）
  - 第 630-648 行：計算 fool rate 並收集成功案例
  - 第 650-722 行：判別器訓練（含動態平衡邏輯）
  - 第 723-726 行：生成器 LoRA SFT（可選）
  - 第 730-799 行：保存結果並計算統計指標

**動態平衡機制：**
- 第 677-687 行：根據 fool rate 決定是否跳過判別器訓練
- 第 1124-1141 行：CLI 參數 `--min-fool-rate-to-train`、`--max-skip-rounds`

**主函數：**
- `main()` 方法（第 1179-1252 行）：解析參數、載入資料、執行多輪訓練、保存歷史

### 自然語言回饋（NL Feedback）

**VAF 回饋建構：** `scripts/gan_training.py`

- **Feedback Block 建構：** 第 556-581 行
  - 第 558 行：檢查 `use_vaf_feedback` 標誌
  - 第 561-569 行：建構檢測結果與信心水準
  - 第 571-573 行：列出檢測原因
  - 第 576 行：標記可疑詞彙
  - 第 580-581 行：提供改進建議

- **Few-shot Cache 建構：** 第 583-599 行
  - 第 584 行：檢查 `use_vaf_fewshot` 標誌與 `successful_examples` 快取
  - 第 590-595 行：插入最佳成功案例作為 few-shot 範例

- **回饋注入：** 第 601-606 行
  - 第 606 行：將建構的回饋 prompt 存入 `example["feedback_prompt"]`
  - 第 500-507 行：在生成器呼叫時將回饋注入 prompt

**回饋生成：** `src/discriminator.py`

- **詳細回饋生成：** `get_detailed_feedback()` 方法（第 266-453 行）
  - 第 305-316 行：計算 P(real) 與信心水準
  - 第 318-358 行：使用注意力機制識別可疑詞彙
  - 第 360-404 行：分析檢測原因（sensationalist_language、vague_attribution、factual_inconsistency、style_mismatch）
  - 第 406-443 行：生成改進建議（基於檢測原因）

**Few-shot Cache 管理：** `scripts/gan_training.py`

- 第 636-648 行：收集成功案例（fooled 且 P(real) > threshold）
- 第 648-653 行：更新 `successful_examples` 快取（最多 5 個）
- 第 213 行：初始化 `self.successful_examples = []`

### 生成器整合（Generator Integration）

**生成器類別：** `src/generator.py` 中的 `FakeNewsGenerator` 類別

- **類別定義：** 第 283-646 行
- **生成方法：** `generate()` 方法（第 318-445 行）
  - 第 325-340 行：建構 prompt（整合 feedback_prompt）
  - 第 342-444 行：呼叫 API 或本地模型生成文本

**Prompt 建構：** `src/generator.py`

- **API Engine：** `APIEngine._build_prompts()` 方法（第 133-200 行）
  - 第 140-148 行：有 feedback 時的 system prompt
  - 第 162-175 行：feedback section 整合
- **Local Engine：** `LocalEngine._build_prompts()` 方法（第 382-500 行）
  - 第 390-410 行：有 feedback 時的 system prompt
  - 第 412-430 行：feedback section 整合

---

## 第四部分：新實驗矩陣（Experiment Matrix）

### 最小可行實驗設計

以下實驗矩陣設計用於系統性地評估 VAF 對假新聞檢測的貢獻，包含 baseline、完整 VAF、以及多個 ablation 變體。

#### 實驗配置

**基礎設定：**
- **資料集：** `sanxing/advfake_news_please`，使用 `train[:5000]` 切片（約 5000 個真實新聞樣本）
- **訓練輪數：** 6 輪（足夠觀察 VAF 效果，同時控制計算成本）
- **判別器：** `microsoft/deberta-v3-base`，每輪 1 epoch
- **生成器：** API 模式（`gpt-4o-mini`）或本地模式（`Qwen/Qwen3-4B-Instruct-2507`）
- **評估資料集：** `sanxing/advfake`（402 個真實/假新聞配對）

**評估指標：**
- Macro-F1、ROC-AUC、Accuracy（在評估資料集上）
- Adversarial Fool Rate（在訓練循環生成的對抗樣本上）
- Detection Reasons Diversity（檢測原因多樣性）

#### 實驗矩陣

| 實驗 ID | VAF Feedback | VAF Few-shot | 描述 | 預期效果 |
|---------|--------------|--------------|------|----------|
| **E1: Baseline** | ❌ | ❌ | 無 VAF，標準 GAN 訓練 | 基準線，檢測器學習基本特徵 |
| **E2: VAF Full** | ✅ | ✅ | 完整 VAF（feedback + few-shot） | 最佳檢測效能與對抗魯棒性 |
| **E3: VAF Feedback Only** | ✅ | ❌ | 僅啟用 feedback block | 評估自然語言回饋的單獨貢獻 |
| **E4: VAF Few-shot Only** | ❌ | ✅ | 僅啟用 few-shot cache | 評估成功案例示範的單獨貢獻 |
| **E5: VAF Weak Feedback** | ⚠️ | ✅ | 弱化 feedback（僅提供檢測結果，無詳細原因） | 評估回饋詳細程度的影響 |
| **E6: VAF Strong Feedback** | ✅✅ | ✅ | 強化 feedback（提供更多檢測原因與建議） | 評估回饋強度的影響 |

#### 實驗執行指令

**E1: Baseline（無 VAF）**

```bash
USE_VAF_FEEDBACK=0 \
USE_VAF_FEWSHOT=0 \
NUM_ROUNDS=6 \
DATASET_SPLIT=train[:5000] \
OUTPUT_DIR=local/runs/exp_baseline_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

**E2: VAF Full（完整 VAF）**

```bash
USE_VAF_FEEDBACK=1 \
USE_VAF_FEWSHOT=1 \
NUM_ROUNDS=6 \
DATASET_SPLIT=train[:5000] \
OUTPUT_DIR=local/runs/exp_vaf_full_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

**E3: VAF Feedback Only**

```bash
USE_VAF_FEEDBACK=1 \
USE_VAF_FEWSHOT=0 \
NUM_ROUNDS=6 \
DATASET_SPLIT=train[:5000] \
OUTPUT_DIR=local/runs/exp_vaf_feedback_only_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

**E4: VAF Few-shot Only**

```bash
USE_VAF_FEEDBACK=0 \
USE_VAF_FEWSHOT=1 \
NUM_ROUNDS=6 \
DATASET_SPLIT=train[:5000] \
OUTPUT_DIR=local/runs/exp_vaf_fewshot_only_$(date +%Y%m%d_%H%M%S) \
./train.sh
```

**E5: VAF Weak Feedback（需修改程式碼）**

此實驗需要修改 `scripts/gan_training.py` 的 VAF 回饋建構邏輯，僅提供檢測結果與信心水準，不提供詳細的檢測原因與改進建議。

**E6: VAF Strong Feedback（需修改程式碼）**

此實驗需要修改 `src/discriminator.py` 的 `get_detailed_feedback()` 方法，增加更多檢測原因類型（如 emotional_manipulation、logical_fallacy）與更詳細的改進建議。

#### 評估腳本

**統一評估所有實驗：**

```bash
# 評估 baseline
python scripts/evaluate_discriminator.py \
  --models-dir local/runs/exp_baseline_YYYYMMDD_HHMMSS \
  --dataset-path local/hf_datasets/advfake/advfake-train.arrow \
  --batch-size 8

# 評估 VAF Full
python scripts/evaluate_discriminator.py \
  --models-dir local/runs/exp_vaf_full_YYYYMMDD_HHMMSS \
  --dataset-path local/hf_datasets/advfake/advfake-train.arrow \
  --batch-size 8

# ... 依此類推
```

**對抗魯棒性評估：**

使用各實驗訓練循環中生成的假新聞樣本（`round_N/` 目錄）作為對抗測試集，評估各實驗的判別器在這些樣本上的表現：

```bash
# 從訓練輸出中提取生成的假新聞樣本
python -c "
from datasets import load_from_disk
import json

for round_id in range(1, 7):
    ds = load_from_disk(f'local/runs/exp_vaf_full_YYYYMMDD_HHMMSS/round_{round_id}')
    # 提取生成的假新聞文本
    fake_texts = [s['text'] for s in ds]
    # 評估判別器在這些樣本上的表現
    # ...
"
```

#### 預期結果分析

**假設：**
- **E2 (VAF Full) > E1 (Baseline)**：完整 VAF 應顯著提升檢測效能與對抗魯棒性
- **E2 > E3, E4**：完整 VAF 應優於單一元件，顯示 feedback 與 few-shot 的協同效應
- **E3 > E4**：Feedback block 的貢獻可能大於 few-shot cache（需實驗驗證）
- **E6 > E2 > E5**：回饋強度與檢測效能應呈正相關（需實驗驗證）

**關鍵指標對比：**
- **Macro-F1 提升：** E2 相較 E1 應有 3-5% 的提升
- **Adversarial Fool Rate 降低：** E2 相較 E1 應有 10-20% 的降低
- **Detection Reasons Diversity：** E2 應顯示更豐富的檢測原因分布

---

## 總結

本文檔完成了從 **Fake News GAN** 到 **Fake News Detection** 的研究重新定位：

1. **現有架構分析**：詳細記錄了當前 GAN 框架的研究問題、方法論、實驗設計與評估指標
2. **Detection 主軸敘事**：重新詮釋 VAF 如何透過語意對齊、生成更難樣本、提升對抗魯棒性來增強假新聞檢測
3. **程式碼對應**：精準標註 Detection 模型、GAN 訓練循環、NL 回饋的實作位置與功能
4. **實驗矩陣**：設計了包含 baseline、完整 VAF、以及多個 ablation 變體的最小可行實驗方案

此重新定位將研究焦點從「生成對抗訓練」轉向「檢測效能提升」，同時保留了 VAF 機制的核心價值：透過自然語言回饋產生更難的對抗樣本，進而提升檢測器的對抗魯棒性與語意理解能力。

