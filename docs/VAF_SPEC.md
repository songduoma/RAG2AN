# VAF (Verbal Adversarial Feedback) 模組規格與重構計劃

本文檔定義 VAF 模組的數學規格、插入點分析、模組化設計與重構計劃。

---

## 第一部分：VAF 插入點分析

### 訓練循環中的插入點

VAF 在 GAN 訓練循環中有三個關鍵插入點：

#### 插入點 1：判別器回饋生成（Discriminator Feedback Generation）

**位置：** `scripts/gan_training.py` 第 540-551 行

**流程：**
```
生成器生成假新聞 → 格式化為判別器輸入 → 判別器評估 → 生成詳細回饋
```

**程式碼對應：**
```python
# 第 531-538 行：格式化假新聞為判別器輸入
fake_disc_input = format_discriminator_input(...)

# 第 540-547 行：判別器評估並生成詳細回饋
self.discriminator.model.eval()
detailed = self.discriminator.get_detailed_feedback(
    fake_disc_input, rag_context=rag_context, top_k=5
)
fake_prob_true = detailed["prob_true"]
```

**輸入：**
- `fake_disc_input`: 格式化的假新聞文本（含可選 RAG 上下文）
- `rag_context`: RAG 檢索上下文（可選）

**輸出：**
- `detailed`: 包含以下欄位的字典：
  - `prob_true`: P(real) 機率
  - `prob_fake`: P(fake) 機率
  - `confidence`: 信心水準（"high"/"medium"/"low"）
  - `suspicious_words`: 可疑詞彙列表 `[(word, attention_score), ...]`
  - `detection_reasons`: 檢測原因列表 `["sensationalist_language", "vague_attribution", ...]`
  - `improvement_suggestions`: 改進建議列表

#### 插入點 2：VAF 回饋建構（VAF Feedback Construction）

**位置：** `scripts/gan_training.py` 第 556-606 行

**流程：**
```
判別器詳細回饋 → 建構 Feedback Block → 建構 Few-shot Block → 組合為完整回饋 Prompt
```

**程式碼對應：**
```python
# 第 556-581 行：建構 Feedback Block
feedback_lines = []
if self.use_vaf_feedback:
    # 建構檢測結果、信心水準、檢測原因、可疑詞彙、改進建議
    feedback_lines.extend([...])

# 第 583-599 行：建構 Few-shot Block
if self.use_vaf_fewshot and self.successful_examples:
    # 插入成功案例作為 few-shot 範例
    feedback_lines.extend([...])

# 第 601-606 行：組合為完整回饋
if feedback_lines:
    example["feedback_prompt"] = "\n".join(feedback_lines)
```

**輸入：**
- `detailed`: 判別器詳細回饋（來自插入點 1）
- `round_id`: 當前訓練輪數
- `self.successful_examples`: 成功案例快取（最多 5 個）

**輸出：**
- `example["feedback_prompt"]`: 完整的自然語言回饋 prompt（字串）

#### 插入點 3：回饋注入生成器（Feedback Injection to Generator）

**位置：** `scripts/gan_training.py` 第 495-507 行

**流程：**
```
回饋 Prompt → 注入生成器 Prompt → 生成器生成新假新聞
```

**程式碼對應：**
```python
# 第 495-499 行：提取回饋（如果存在）
feedback = (
    example.get("feedback_prompt")
    if (self.use_vaf_feedback or self.use_vaf_fewshot)
    else None
)

# 第 500-507 行：將回饋注入生成器
gen_output = self.generator.generate(
    content=example.get("description", ""),
    feedback_prompt=feedback,  # ← VAF 回饋注入點
    use_rag=use_rag,
    ...
)
```

**生成器整合：** `src/generator.py`
- **API Engine：** 第 133-200 行（`_build_prompts()` 方法）
- **Local Engine：** 第 331-414 行（`_build_prompts()` 方法）
- 當 `feedback_prompt` 不為 `None` 時，會調整 system prompt 並在 user prompt 中插入 feedback section

### 回饋流程圖

```
┌─────────────────────────────────────────────────────────────┐
│                    GAN Training Round                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  1. Generator generates fake news    │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  2. Discriminator evaluates fake      │
        │     → get_detailed_feedback()          │
        │     [INSERTION POINT 1]                │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  3. Build VAF feedback prompt          │
        │     - Feedback Block (if enabled)      │
        │     - Few-shot Block (if enabled)      │
        │     [INSERTION POINT 2]                │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  4. Inject feedback into generator     │
        │     → generator.generate(feedback=...) │
        │     [INSERTION POINT 3]                │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  5. Next round: Generator uses        │
        │     feedback to improve generation    │
        └───────────────────────────────────────┘
```

### Few-shot Cache 管理

**位置：** `scripts/gan_training.py` 第 630-671 行

**流程：**
```
每輪結束 → 計算 fool rate → 收集成功案例 → 更新 few-shot cache
```

**成功案例條件：**
- `label == 0`（假新聞）
- `fool == True`（P(real) > 0.5，成功騙過判別器）
- `prob_true > threshold`（預設 0.55，可透過 `--gen-sft-success-threshold` 調整）

**Cache 限制：**
- 最多保留 5 個成功案例（第 668 行）
- 按 `prob_true` 降序排序，保留最佳案例

---

## 第二部分：數學/偽代碼規格

### 輸入輸出規格

#### 輸入（Input）

**判別器評估階段：**
- `x_fake`: 生成的假新聞文本（字串）
- `x_real`: 對應的真實新聞文本（字串，用於 RAG 檢索）
- `θ_D`: 判別器參數（DeBERTa 模型權重）
- `rag_context`: RAG 檢索上下文（可選，字串）

**VAF 回饋建構階段：**
- `detailed`: 判別器詳細回饋（字典）
- `round_id`: 當前訓練輪數（整數）
- `S`: 成功案例快取（列表，最多 5 個元素）
- `α`: 回饋強度係數（浮點數，0.0-1.0，預設 1.0）

#### 輸出（Output）

**VAF 回饋 Prompt：**
- `f_prompt`: 完整的自然語言回饋 prompt（字串）
  - 包含 Feedback Block（如果啟用）
  - 包含 Few-shot Block（如果啟用）

### 計算流程（Computation Flow）

#### 步驟 1：判別器詳細回饋生成

```python
def get_detailed_feedback(discriminator, x_fake, rag_context=""):
    """
    輸入:
        discriminator: EncoderDiscriminator 實例
        x_fake: 假新聞文本
        rag_context: RAG 上下文（可選）
    
    輸出:
        detailed: {
            prob_true: float,           # P(real)
            prob_fake: float,           # P(fake) = 1 - P(real)
            confidence: str,             # "high" | "medium" | "low"
            suspicious_words: List[Tuple[str, float]],  # [(word, attention_score), ...]
            detection_reasons: List[str], # ["sensationalist_language", ...]
            improvement_suggestions: List[str]  # ["Replace ...", ...]
        }
    """
    # 1. Tokenize 輸入
    inputs = tokenizer(x_fake, return_tensors="pt")
    
    # 2. 前向傳播（含注意力）
    outputs = discriminator.model(**inputs, output_attentions=True)
    
    # 3. 計算機率
    probs = softmax(outputs.logits, dim=-1)
    prob_true = probs[0, positive_label_id]
    prob_fake = 1 - prob_true
    
    # 4. 計算信心水準
    if prob_fake >= 0.8:
        confidence = "high"
    elif prob_fake >= 0.6:
        confidence = "medium"
    else:
        confidence = "low"
    
    # 5. 提取可疑詞彙（使用注意力機制）
    attentions = outputs.attentions[-1]  # 最後一層注意力
    avg_att = mean(attentions, dim=1)    # 平均所有 head
    cls_att = avg_att[0, :]              # [CLS] 對其他 token 的注意力
    suspicious_words = top_k(cls_att, k=5)  # Top-K 可疑詞彙
    
    # 6. 分析檢測原因
    detection_reasons = analyze_detection_reasons(
        suspicious_words, prob_fake
    )
    
    # 7. 生成改進建議
    improvement_suggestions = generate_suggestions(
        detection_reasons, suspicious_words
    )
    
    return {
        "prob_true": prob_true,
        "prob_fake": prob_fake,
        "confidence": confidence,
        "suspicious_words": suspicious_words,
        "detection_reasons": detection_reasons,
        "improvement_suggestions": improvement_suggestions
    }
```

#### 步驟 2：VAF 回饋建構

```python
def build_vaf_feedback(
    detailed, round_id, successful_examples, 
    feedback_enabled=True, fewshot_enabled=True, alpha=1.0
):
    """
    輸入:
        detailed: 判別器詳細回饋
        round_id: 當前訓練輪數
        successful_examples: 成功案例快取
        feedback_enabled: 是否啟用 Feedback Block
        fewshot_enabled: 是否啟用 Few-shot Block
        alpha: 回饋強度係數（0.0-1.0）
    
    輸出:
        f_prompt: 完整的 VAF 回饋 prompt（字串）
    """
    feedback_lines = []
    
    # === Feedback Block ===
    if feedback_enabled and alpha > 0:
        predicted_label = "REAL" if detailed["prob_true"] > 0.5 else "FAKE"
        
        # 基礎回饋（不受 alpha 影響）
        feedback_lines.extend([
            f"=== DISCRIMINATOR FEEDBACK (Round {round_id}) ===",
            f"Detection Result: {predicted_label}",
            f"Confidence: {detailed['confidence'].upper()}",
            ""
        ])
        
        # 強度調整：根據 alpha 決定包含多少詳細資訊
        if alpha >= 0.5:
            # 包含檢測原因
            feedback_lines.append("Problems Identified:")
            for reason in detailed["detection_reasons"]:
                feedback_lines.append(f"  - {reason}")
            feedback_lines.append("")
        
        if alpha >= 0.7:
            # 包含可疑詞彙
            suspicious = ", ".join([
                f"{w}({s:.2f})" 
                for w, s in detailed["suspicious_words"][:int(3 * alpha)]
            ])
            feedback_lines.append(f"Flagged Suspicious Terms: {suspicious}")
            feedback_lines.append("")
        
        if alpha >= 0.9:
            # 包含完整改進建議
            feedback_lines.append("Improvement Instructions:")
            for suggestion in detailed["improvement_suggestions"]:
                feedback_lines.append(f"  - {suggestion}")
    
    # === Few-shot Block ===
    if fewshot_enabled and successful_examples:
        if feedback_lines:
            feedback_lines.append("")
        feedback_lines.extend([
            "=" * 50,
            "SUCCESSFUL EXAMPLE (This fooled the detector!):",
            "=" * 50
        ])
        best_example = successful_examples[-1]
        feedback_lines.append(best_example["text"][:500])
        feedback_lines.append(f"[This achieved {best_example['prob_true']:.1%} real probability]")
        feedback_lines.append("LEARN FROM THIS: Mimic the style and tone above.")
    
    # === 組合為完整 Prompt ===
    if feedback_lines:
        feedback_lines.append("")
        feedback_lines.append("CRITICAL: Your rewrite MUST address these issues.")
        f_prompt = "\n".join(feedback_lines)
    else:
        f_prompt = None
    
    return f_prompt
```

#### 步驟 3：回饋注入生成器

```python
def inject_feedback_to_generator(generator, x_real, f_prompt, use_rag=False):
    """
    輸入:
        generator: FakeNewsGenerator 實例
        x_real: 真實新聞文本
        f_prompt: VAF 回饋 prompt（可為 None）
        use_rag: 是否使用 RAG
    
    輸出:
        x_fake_new: 根據回饋改進後的新假新聞文本
    """
    if f_prompt:
        # 有回饋：調整 system prompt 為對抗模式
        system_prompt = (
            "You are a sophisticated writer engaged in an adversarial game. "
            "Your previous attempt was DETECTED. You must learn from feedback."
        )
        user_prompt = f"""
        Original Real News:
        {x_real}
        
        {f_prompt}
        
        Task: Rewrite to be fake but realistic, addressing the feedback above.
        """
    else:
        # 無回饋：標準生成模式
        system_prompt = "You are a sophisticated writer. Rewrite news to be fake."
        user_prompt = f"Original: {x_real}\nTask: Rewrite to be fake."
    
    x_fake_new = generator.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        use_rag=use_rag
    )
    
    return x_fake_new
```

### 對 Loss/Gradient 的影響

**重要：** VAF 不直接影響 loss 或 gradient 計算，而是間接影響生成器輸出品質，進而影響判別器的訓練。

#### 數學表述

**標準 GAN Loss（無 VAF）：**
```
L_D = -E[log D(x_real)] - E[log(1 - D(G(x_real)))]
L_G = -E[log D(G(x_real))]
```

**VAF 增強流程：**
1. **第 t 輪：**
   - 生成器生成：`x_fake^t = G(x_real, f_prompt^{t-1})`
   - 判別器評估：`D(x_fake^t)` → 生成 `f_prompt^t`
   - 判別器訓練：`L_D^t = -E[log D(x_real)] - E[log(1 - D(x_fake^t))]`

2. **第 t+1 輪：**
   - 生成器使用回饋：`x_fake^{t+1} = G(x_real, f_prompt^t)`
   - 由於 `f_prompt^t` 包含改進建議，`x_fake^{t+1}` 品質提升
   - 判別器面對更難樣本：`L_D^{t+1} = -E[log D(x_real)] - E[log(1 - D(x_fake^{t+1}))]`

**間接影響：**
- VAF 透過提升 `x_fake` 的品質（更難檢測），迫使判別器學習更魯棒的特徵
- 這反映在判別器 loss 的變化：`L_D^{t+1}` 可能比 `L_D^t` 更高（因為樣本更難）
- 但長期來看，判別器會適應並改進，loss 會下降

**Gradient Flow：**
```
VAF 回饋 → 生成器 prompt → 生成器輸出 x_fake
                                        ↓
                              判別器輸入 x_fake
                                        ↓
                              判別器 loss L_D
                                        ↓
                              判別器 gradient ∇_θ_D L_D
```

VAF 不直接參與梯度計算，但影響生成器輸出，進而影響判別器的梯度。

---

## 第三部分：VAF 模組抽象設計

### 模組介面設計

```python
class VAFModule:
    """
    Verbal Adversarial Feedback (VAF) 模組
    
    將 VAF 回饋生成與應用邏輯封裝為可切換的模組。
    """
    
    def __init__(
        self,
        enabled: bool = True,
        feedback_enabled: bool = True,
        fewshot_enabled: bool = True,
        alpha: float = 1.0,
        template: Optional[str] = None,
        max_fewshot_examples: int = 5,
        success_threshold: float = 0.55,
    ):
        """
        參數:
            enabled: 是否啟用 VAF 模組（總開關）
            feedback_enabled: 是否啟用 Feedback Block
            fewshot_enabled: 是否啟用 Few-shot Block
            alpha: 回饋強度係數（0.0-1.0）
            template: 自訂回饋模板（可選）
            max_fewshot_examples: 最多保留的成功案例數
            success_threshold: 成功案例的 P(real) 閾值
        """
        self.enabled = enabled
        self.feedback_enabled = feedback_enabled and enabled
        self.fewshot_enabled = fewshot_enabled and enabled
        self.alpha = alpha
        self.template = template
        self.max_fewshot_examples = max_fewshot_examples
        self.success_threshold = success_threshold
        
        self.successful_examples = []  # Few-shot cache
    
    def generate_feedback(
        self,
        discriminator,
        fake_text: str,
        round_id: int,
        rag_context: str = "",
    ) -> Optional[str]:
        """
        生成 VAF 回饋 prompt
        
        輸入:
            discriminator: EncoderDiscriminator 實例
            fake_text: 生成的假新聞文本
            round_id: 當前訓練輪數
            rag_context: RAG 上下文（可選）
        
        輸出:
            feedback_prompt: VAF 回饋 prompt（如果啟用），否則 None
        """
        if not self.enabled:
            return None
        
        # 1. 獲取判別器詳細回饋
        detailed = discriminator.get_detailed_feedback(
            fake_text, rag_context=rag_context, top_k=5
        )
        
        # 2. 建構回饋 prompt
        feedback_prompt = self._build_feedback_prompt(
            detailed, round_id
        )
        
        return feedback_prompt
    
    def _build_feedback_prompt(
        self,
        detailed: dict,
        round_id: int,
    ) -> Optional[str]:
        """建構回饋 prompt（內部方法）"""
        feedback_lines = []
        
        # Feedback Block
        if self.feedback_enabled:
            feedback_lines.extend(
                self._build_feedback_block(detailed, round_id)
            )
        
        # Few-shot Block
        if self.fewshot_enabled and self.successful_examples:
            if feedback_lines:
                feedback_lines.append("")
            feedback_lines.extend(
                self._build_fewshot_block()
            )
        
        # 組合
        if feedback_lines:
            feedback_lines.append("")
            feedback_lines.append(
                "CRITICAL: Your rewrite MUST address these issues."
            )
            return "\n".join(feedback_lines)
        
        return None
    
    def _build_feedback_block(
        self, detailed: dict, round_id: int
    ) -> List[str]:
        """建構 Feedback Block（內部方法）"""
        lines = []
        predicted_label = (
            "REAL" if detailed["prob_true"] > 0.5 else "FAKE"
        )
        
        # 基礎資訊（不受 alpha 影響）
        lines.extend([
            f"=== DISCRIMINATOR FEEDBACK (Round {round_id}) ===",
            f"Detection Result: {predicted_label}",
            f"Confidence: {detailed['confidence'].upper()} "
            f"({detailed['prob_true']:.1%} real / "
            f"{detailed['prob_fake']:.1%} fake)",
            "",
        ])
        
        # 根據 alpha 決定詳細程度
        if self.alpha >= 0.5:
            lines.append("Problems Identified:")
            for reason in detailed["detection_reasons"]:
                reason_readable = reason.replace("_", " ").title()
                lines.append(f"  - {reason_readable}")
            lines.append("")
        
        if self.alpha >= 0.7:
            suspicious = ", ".join([
                f"{w}({s:.2f})"
                for w, s in detailed["suspicious_words"][
                    :int(3 * self.alpha)
                ]
            ])
            lines.append(f"Flagged Suspicious Terms: {suspicious}")
            lines.append("")
        
        if self.alpha >= 0.9:
            lines.append("Improvement Instructions:")
            for suggestion in detailed["improvement_suggestions"]:
                lines.append(f"  - {suggestion}")
        
        return lines
    
    def _build_fewshot_block(self) -> List[str]:
        """建構 Few-shot Block（內部方法）"""
        lines = [
            "=" * 50,
            "SUCCESSFUL EXAMPLE (This fooled the detector!):",
            "=" * 50,
        ]
        best_example = self.successful_examples[-1]
        lines.append(best_example["text"][:500])
        lines.append("...")
        lines.append(
            f"[This achieved {best_example['prob_true']:.1%} real probability]"
        )
        lines.append("")
        lines.append(
            "LEARN FROM THIS: Mimic the style and tone of the successful example above."
        )
        return lines
    
    def update_fewshot_cache(
        self, fake_samples: List[dict]
    ) -> None:
        """
        更新 Few-shot cache
        
        輸入:
            fake_samples: 本輪生成的假新聞樣本列表
        """
        if not self.fewshot_enabled:
            return
        
        # 收集成功案例
        new_successes = [
            s
            for s in fake_samples
            if s.get("label", 1) == 0
            and s.get("fool", False)
            and s.get("prob_true", 0.0) > self.success_threshold
        ]
        
        if new_successes:
            # 按 prob_true 排序
            new_successes.sort(key=lambda x: x["prob_true"], reverse=True)
            
            # 更新 cache（最多保留 max_fewshot_examples 個）
            for s in new_successes[:3]:  # 每輪最多加 3 個
                self.successful_examples.append({
                    "text": s.get("generated_content", s["text"][:500]),
                    "prob_true": s["prob_true"],
                })
            
            # 限制 cache 大小
            self.successful_examples = self.successful_examples[
                -self.max_fewshot_examples:
            ]
```

### 使用範例

```python
# 初始化 VAF 模組
vaf = VAFModule(
    enabled=True,
    feedback_enabled=True,
    fewshot_enabled=True,
    alpha=1.0,
)

# 在訓練循環中使用
for round_id in range(1, num_rounds + 1):
    for example in generator_dataset:
        # 生成假新聞（第一輪無回饋）
        feedback = example.get("feedback_prompt")
        fake = generator.generate(content=..., feedback_prompt=feedback)
        
        # VAF 生成回饋（用於下一輪）
        feedback_prompt = vaf.generate_feedback(
            discriminator=discriminator,
            fake_text=fake,
            round_id=round_id,
        )
        example["feedback_prompt"] = feedback_prompt
    
    # 更新 few-shot cache
    vaf.update_fewshot_cache(fake_samples)
```

---

## 第四部分：新增配置參數

### CLI 參數清單

| 參數名稱 | 類型 | 預設值 | 說明 |
|---------|------|--------|------|
| `--vaf-enabled` | bool | `True` | VAF 模組總開關 |
| `--disable-vaf` | flag | - | 等同 `--vaf-enabled=False` |
| `--vaf-feedback-enabled` | bool | `True` | 啟用 Feedback Block |
| `--disable-nl-feedback` | flag | - | 等同 `--vaf-feedback-enabled=False` |
| `--vaf-fewshot-enabled` | bool | `True` | 啟用 Few-shot Block |
| `--vaf-alpha` | float | `1.0` | 回饋強度係數（0.0-1.0） |
| `--vaf-max-fewshot` | int | `5` | 最多保留的成功案例數 |
| `--vaf-success-threshold` | float | `0.55` | 成功案例的 P(real) 閾值 |
| `--vaf-template` | str | `None` | 自訂回饋模板（可選） |

### 環境變數對應

| 環境變數 | CLI 參數 | 預設值 |
|---------|---------|--------|
| `VAF_ENABLED` | `--vaf-enabled` | `1` |
| `VAF_FEEDBACK_ENABLED` | `--vaf-feedback-enabled` | `1` |
| `VAF_FEWSHOT_ENABLED` | `--vaf-fewshot-enabled` | `1` |
| `VAF_ALPHA` | `--vaf-alpha` | `1.0` |
| `VAF_MAX_FEWSHOT` | `--vaf-max-fewshot` | `5` |
| `VAF_SUCCESS_THRESHOLD` | `--vaf-success-threshold` | `0.55` |

### 向後相容性

**保留現有參數：**
- `--use-vaf-feedback` / `--no-vaf-feedback` → 映射到 `--vaf-feedback-enabled`
- `--use-vaf-fewshot` / `--no-vaf-fewshot` → 映射到 `--vaf-fewshot-enabled`
- `USE_VAF_FEEDBACK` / `USE_VAF_FEWSHOT` → 映射到對應的新環境變數

**預設行為：**
- 如果未指定新參數，使用現有參數的值
- 如果新舊參數衝突，新參數優先

---

## 第五部分：最小改動重構計劃

### 檔案修改清單

#### 1. 新增檔案：`src/vaf.py`

**內容：** VAF 模組實作（見第三部分）

**功能：**
- `VAFModule` 類別
- `generate_feedback()` 方法
- `update_fewshot_cache()` 方法
- 內部方法：`_build_feedback_prompt()`, `_build_feedback_block()`, `_build_fewshot_block()`

#### 2. 修改檔案：`scripts/gan_training.py`

**修改點 1：導入 VAF 模組（第 1-32 行附近）**
```python
# 新增
from src.vaf import VAFModule
```

**修改點 2：GANTrainer.__init__()（第 110-213 行）**
```python
# 舊程式碼：
self.use_vaf_feedback = getattr(args, "use_vaf_feedback", True)
self.use_vaf_fewshot = getattr(args, "use_vaf_fewshot", True)
self.successful_examples = []

# 新程式碼：
# 初始化 VAF 模組
self.vaf = VAFModule(
    enabled=getattr(args, "vaf_enabled", True),
    feedback_enabled=getattr(args, "vaf_feedback_enabled", 
        getattr(args, "use_vaf_feedback", True)),  # 向後相容
    fewshot_enabled=getattr(args, "vaf_fewshot_enabled",
        getattr(args, "use_vaf_fewshot", True)),  # 向後相容
    alpha=getattr(args, "vaf_alpha", 1.0),
    max_fewshot_examples=getattr(args, "vaf_max_fewshot", 5),
    success_threshold=getattr(args, "vaf_success_threshold", 0.55),
)
```

**修改點 3：run_round() - 回饋生成（第 540-606 行）**
```python
# 舊程式碼（第 540-606 行）：
# 使用 VAF 模組替換手動建構邏輯

# 新程式碼：
# 使用 VAF 模組生成回饋
feedback_prompt = self.vaf.generate_feedback(
    discriminator=self.discriminator,
    fake_text=fake_disc_input,
    round_id=round_id,
    rag_context=rag_context if self.vaf.feedback_enabled else "",
)
example["feedback_prompt"] = feedback_prompt
```

**修改點 4：run_round() - Few-shot Cache 更新（第 630-671 行）**
```python
# 舊程式碼（第 636-671 行）：
# 手動管理 successful_examples

# 新程式碼：
# 使用 VAF 模組更新 cache
self.vaf.update_fewshot_cache(fake_samples)
```

**修改點 5：parse_args() - 新增參數（第 1148-1173 行後）**
```python
# 新增 VAF 參數
parser.add_argument(
    "--vaf-enabled",
    action="store_true",
    default=os.environ.get("VAF_ENABLED", "1").lower() in ("1", "true", "yes"),
    help="Enable VAF module (master switch).",
)
parser.add_argument(
    "--disable-vaf",
    dest="vaf_enabled",
    action="store_false",
    help="Disable VAF module entirely.",
)
parser.add_argument(
    "--vaf-feedback-enabled",
    action="store_true",
    default=os.environ.get("VAF_FEEDBACK_ENABLED", "1").lower() in ("1", "true", "yes"),
    help="Enable VAF feedback block.",
)
parser.add_argument(
    "--disable-nl-feedback",
    dest="vaf_feedback_enabled",
    action="store_false",
    help="Disable NL feedback block (keep few-shot if enabled).",
)
parser.add_argument(
    "--vaf-fewshot-enabled",
    action="store_true",
    default=os.environ.get("VAF_FEWSHOT_ENABLED", "1").lower() in ("1", "true", "yes"),
    help="Enable VAF few-shot cache.",
)
parser.add_argument(
    "--vaf-alpha",
    type=float,
    default=float(os.environ.get("VAF_ALPHA", "1.0")),
    help="VAF feedback intensity (0.0 = no effect, 1.0 = full effect).",
)
parser.add_argument(
    "--vaf-max-fewshot",
    type=int,
    default=int(os.environ.get("VAF_MAX_FEWSHOT", "5")),
    help="Maximum number of successful examples in few-shot cache.",
)
parser.add_argument(
    "--vaf-success-threshold",
    type=float,
    default=float(os.environ.get("VAF_SUCCESS_THRESHOLD", "0.55")),
    help="P(real) threshold for successful examples.",
)

# 保留舊參數（向後相容）
# --use-vaf-feedback / --no-vaf-feedback (已存在)
# --use-vaf-fewshot / --no-vaf-fewshot (已存在)
```

#### 3. 修改檔案：`train.sh`

**修改點：新增環境變數（第 60-66 行附近）**
```bash
# --- VAF 設定 ---
VAF_ENABLED="${VAF_ENABLED:-1}"
VAF_FEEDBACK_ENABLED="${VAF_FEEDBACK_ENABLED:-${USE_VAF_FEEDBACK:-1}}"
VAF_FEWSHOT_ENABLED="${VAF_FEWSHOT_ENABLED:-${USE_VAF_FEWSHOT:-1}}"
VAF_ALPHA="${VAF_ALPHA:-1.0}"
VAF_MAX_FEWSHOT="${VAF_MAX_FEWSHOT:-5}"
VAF_SUCCESS_THRESHOLD="${VAF_SUCCESS_THRESHOLD:-0.55}"

# 匯出環境變數
export VAF_ENABLED="$VAF_ENABLED"
export VAF_FEEDBACK_ENABLED="$VAF_FEEDBACK_ENABLED"
export VAF_FEWSHOT_ENABLED="$VAF_FEWSHOT_ENABLED"
export VAF_ALPHA="$VAF_ALPHA"
export VAF_MAX_FEWSHOT="$VAF_MAX_FEWSHOT"
export VAF_SUCCESS_THRESHOLD="$VAF_SUCCESS_THRESHOLD"
```

**修改點：傳遞參數到 Python 腳本（第 132-169 行）**
```bash
python -u scripts/gan_training.py \
  ... \
  $( [[ "$VAF_ENABLED" == "1" ]] && echo "--vaf-enabled" || echo "--disable-vaf" ) \
  $( [[ "$VAF_FEEDBACK_ENABLED" == "1" ]] && echo "--vaf-feedback-enabled" || echo "--disable-nl-feedback" ) \
  $( [[ "$VAF_FEWSHOT_ENABLED" == "1" ]] && echo "--vaf-fewshot-enabled" || echo "--no-vaf-fewshot" ) \
  --vaf-alpha "$VAF_ALPHA" \
  --vaf-max-fewshot "$VAF_MAX_FEWSHOT" \
  --vaf-success-threshold "$VAF_SUCCESS_THRESHOLD" \
  ...
```

### 向後相容性保證

**策略：**
1. 保留所有現有 CLI 參數與環境變數
2. 在 `GANTrainer.__init__()` 中，優先使用新參數，但支援舊參數作為 fallback
3. 確保預設行為與現有實作一致

**測試檢查清單：**
- [ ] 使用舊參數 `--use-vaf-feedback` 仍能正常運作
- [ ] 使用舊環境變數 `USE_VAF_FEEDBACK=0` 仍能正常運作
- [ ] 預設行為（無指定參數）與現有實作一致
- [ ] 新參數 `--vaf-alpha=0.5` 能正確調整回饋強度
- [ ] `--disable-vaf` 能完全停用 VAF

### 不破壞現有結果的保證

**策略：**
1. **預設值一致：** 所有新參數的預設值與現有實作的行為一致
2. **邏輯等價：** VAF 模組的邏輯與現有實作等價（僅封裝，不改變計算）
3. **漸進式遷移：** 先實作模組，再逐步替換現有邏輯

**驗證步驟：**
1. 使用相同參數執行現有版本與新版本
2. 比較 `training_history.json` 中的指標（fool rate, mean P(real), disc loss）
3. 確保差異在數值誤差範圍內（< 0.1%）

---

## 第六部分：Ablation Flags 設計

### Flag 清單與對應行為

| Flag | 等價參數 | 行為 |
|------|---------|------|
| `--disable-vaf` | `--vaf-enabled=False` | 完全停用 VAF（feedback + few-shot） |
| `--disable-nl-feedback` | `--vaf-feedback-enabled=False` | 停用 NL feedback，保留 few-shot（如果啟用） |
| `--vaf-alpha=0.0` | - | 回饋強度為 0（僅提供基礎檢測結果） |
| `--vaf-alpha=0.5` | - | 中等強度（包含檢測原因，不含可疑詞彙） |
| `--vaf-alpha=1.0` | - | 完整強度（包含所有詳細資訊） |

### 使用範例

```bash
# 完全停用 VAF
./train.sh --disable-vaf

# 停用 NL feedback，保留 few-shot
./train.sh --disable-nl-feedback

# 使用弱回饋（alpha=0.5）
./train.sh --vaf-alpha=0.5

# 組合使用
./train.sh --disable-nl-feedback --vaf-fewshot-enabled
```

### 環境變數對應

```bash
# 完全停用
export VAF_ENABLED=0
./train.sh

# 停用 NL feedback
export VAF_FEEDBACK_ENABLED=0
./train.sh

# 調整回饋強度
export VAF_ALPHA=0.5
./train.sh
```

---

## 總結

本文檔完成了 VAF 模組化的完整規格：

1. **插入點分析：** 識別了三個關鍵插入點與回饋流程
2. **數學規格：** 定義了輸入輸出、計算流程與對 loss/gradient 的間接影響
3. **模組抽象：** 設計了 `VAFModule` 類別，支援可配置參數
4. **配置參數：** 列出了所有新參數與預設值
5. **重構計劃：** 提供了最小改動的實作計劃，確保向後相容
6. **Ablation Flags：** 設計了 `--disable-vaf`、`--disable-nl-feedback`、`--vaf-alpha` 等 flags

此模組化設計將 VAF 邏輯封裝為獨立模組，便於維護、測試與擴展，同時保持與現有實作的相容性。

