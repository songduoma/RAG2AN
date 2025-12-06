# Generator 格式問題修正

## 問題描述

訓練時發現生成器的 fool rate 一直是 0%，Mean P(real) 接近 0.5（隨機猜測水平）。

### 根本原因

**不匹配的格式期望**：

1. `gan_training.py` 中的 `extract_article()` 函數期望生成的文本包含 `Title:` 和 `Body:` 標籤
2. 但 `generator.py` 的 prompt 明確要求**不要**使用這些標籤，直接輸出新聞內容
3. 結果：`extract_article()` 找不到標籤，返回空標題或格式錯誤的內容

## 修正方案

### 1. 更新 Generator Prompts

**修改位置**：`generator.py` 中的 `APIEngine._build_prompts()` 和 `LocalEngine._build_prompts()`

**System Prompt 變更**：
```python
# Before
"IMPORTANT: Do NOT use Markdown formatting (no ###, no **, no bullet points). "
"Keep the same paragraph structure and spacing as the original news article."

# After
"CRITICAL FORMAT: You MUST output in 'Title:' and 'Body:' format. "
"The Title should be a single line. The Body should follow on a new line after 'Body:'."
```

**User Prompt 變更**：
```python
# Before
"CRITICAL FORMATTING RULES:
1. Do NOT use any Markdown formatting (no ###, no **, no bullet points)
2. Do NOT add extra blank lines between paragraphs
3. Keep the EXACT same paragraph structure as the original
4. Start directly with the news content (e.g., \"(CNN)...\" or similar)
5. Output ONLY the rewritten fake news article, nothing else"

# After
"CRITICAL FORMATTING RULES:
1. Output in this EXACT format:
   Title: [Your rewritten title here]
   Body: [Your rewritten article here]
2. Do NOT use Markdown formatting in the body (no ###, no **, no bullet points)
3. Keep the same paragraph structure as the original in the Body section
4. The Title should be a single line
5. The Body should start on a new line after \"Body:\""
```

### 2. 改進 extract_article() 函數

**修改位置**：`gan_training.py`

**新功能**：
- 支援顯式 `Title:` / `Body:` 格式
- **容錯處理**：如果模型不遵循格式，自動將第一行當作標題，其餘當作正文

```python
def extract_article(text: str) -> Tuple[str, str]:
    """
    Extract generated Title/Body from the generator output.
    
    Handles multiple formats:
    1. Explicit format: "Title: ...\nBody: ..."
    2. Implicit format: First line as title, rest as body
    """
    if not isinstance(text, str) or not text.strip():
        return "", ""
    
    text = text.strip()
    
    # Case 1: Explicit Title:/Body: format
    if "Title:" in text:
        parts = text.split("Title:", 1)[1]
        if "Body:" in parts:
            title_part, body_part = parts.split("Body:", 1)
            title = title_part.strip().splitlines()[0]
            body = body_part.strip()
            return title, body
        else:
            return "", parts.strip()
    
    # Case 2: No explicit markers - use first line as title
    lines = text.split('\n', 1)
    if len(lines) == 1:
        return "", lines[0].strip()
    else:
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        return title, body
```

## 測試結果

### Before Fix
```
Mean P(real): 0.473 → 0.489 (接近隨機)
Fool Rate:    0.0% → 0.0% (完全失敗)
```

### After Fix
```
Round 1:
  Mean P(real): 0.506
  Fool Rate:    100.0% ✅
  
Round 2:
  Mean P(real): 0.496
  Fool Rate:    0.0% (判別器變強，正常的對抗過程)
```

### 生成內容範例

**Round 1 輸出**：
```
Title: Daniel Radcliffe's £20 Million Fortune Now at His Disposal as He Turns 19
Body: LONDON, England (Reuters) -- Daniel Radcliffe, the beloved star of the Harry Potter franchise, is set to inherit a staggering £20 million fortune upon turning 19 on Monday...
```

**提取結果**：
```
Extracted Title: ✅ Daniel Radcliffe's £20 Million Fortune Now at His Disposal as He Turns 19
Extracted Body: ✅ LONDON, England (Reuters) -- Daniel Radcliffe, the beloved star...
```

## 關鍵學習

1. **Prompt 和解析必須一致**：Generator 的輸出格式必須與下游處理邏輯匹配
2. **容錯處理很重要**：LLM 不一定會完全遵循指令，需要智能的 fallback 機制
3. **測試小樣本**：使用 `train[:2]` 快速驗證修正，節省 GPU 時間

## 使用方法

修正後的訓練命令：
```bash
export GEN_MODE=local
export CUDA_VISIBLE_DEVICES=1
DATASET_SPLIT='train[:50]' NUM_ROUNDS=5 BATCH_SIZE=4 ./train.sh
```

或使用 API 模式（推薦）：
```bash
export GEN_MODE=api
export OPENAI_API_KEY=sk-xxx
export OPENAI_MODEL=gpt-4o-mini
DATASET_SPLIT='train[:50]' NUM_ROUNDS=5 BATCH_SIZE=4 ./train.sh
```
