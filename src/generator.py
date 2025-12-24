"""
RAG²AN Generator - 支援 API 和本地模型兩種模式

使用方式：
    API 模式（推薦，不需 GPU）：
        export GEN_MODE=api
        export OPENAI_API_KEY=sk-xxx
        export OPENAI_MODEL=gpt-4o-mini  # 可選，預設 gpt-4o-mini

    本地模式（需要 GPU）：
        export GEN_MODE=local
        # 會自動載入 Qwen/Qwen2.5-7B-Instruct
"""

import os
import requests
from collections import Counter
from typing import Dict, Optional, Union

# ==========================================
# 1. 配置設定 (Configuration)
# ==========================================

# 模式選擇：'api' 或 'local'
GEN_MODE = os.environ.get("GEN_MODE", "local")
GEN_USE_LORA = os.environ.get("GEN_USE_LORA", "true").lower() in ("true", "1", "yes")
GEN_LORA_R = int(os.environ.get("GEN_LORA_R", "8"))
GEN_LORA_ALPHA = int(os.environ.get("GEN_LORA_ALPHA", "16"))
GEN_LORA_DROPOUT = float(os.environ.get("GEN_LORA_DROPOUT", "0.05"))
GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "512"))

# API 設定（OpenAI 相容）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 本地模型設定
LOCAL_MODEL_ID = os.environ.get("GEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

# 為了向後相容，保留 MODEL_ID
MODEL_ID = LOCAL_MODEL_ID

FACT_CHANGE_INSTRUCTION = (
    "CRITICAL: You MUST introduce 1–2 factual changes that meaningfully alter the story while keeping the SAME main event. "
    "The article must remain about the exact same central event as the original (e.g., the same speech, trial, accident, or policy decision). "
    "Do NOT switch to a different event, topic, organization, or country; keep the same core incident and storyline. "
    "In the VERY FIRST sentence of the rewritten article, you MUST change at least one core fact: WHO, WHERE, WHEN, a KEY NUMBER, or the OUTCOME. "
    "You must NOT copy the first sentence verbatim; its wording and at least one core fact must be different from the original. "
    "You may change who was involved, where it happened, when it happened, important numbers (amounts, years, percentages), or the cause/outcome of events. "
    "Do not rely solely on paraphrasing. At least one factual element must change. "
    "Do NOT invent a completely new, unrelated second event (such as a new protest, scandal, or case) that is not a direct variation of the original main event. "
    "All changes must remain coherent with each other and with the (modified) main event."
)

LENGTH_CONSTRAINT = (
    "Keep the rewritten article roughly within 80%–120% of the original length; do not make it significantly longer or shorter. "
)

STYLE_CONSTRAINT = (
    "Write in a neutral, professional journalistic tone, avoid sensationalism, and do NOT use Markdown "
    "formatting (no ###, no **, no bullet points). Keep the same paragraph structure and spacing as the original article. "
)

def _compute_max_new_tokens(content: str, tokenizer=None, global_cap: int = GEN_MAX_NEW_TOKENS) -> int:
    """
    Heuristic: keep generation close to the original length.
    Uses tokenizer length when available; otherwise estimates from word count.
    """
    if tokenizer is not None:
        ids = tokenizer(content, add_special_tokens=False)["input_ids"]
        base_len = len(ids)
    else:
        # crude estimate: ~1.3 tokens per word
        base_len = int(len(content.split()) * 1.3)
    target = int(base_len * 1.15) + 5  # allow slight growth but avoid doubling
    target = max(32, target)  # ensure not too small
    return min(global_cap, target)


def _format_date(date_val: object) -> str:
    if hasattr(date_val, "date"):
        return str(date_val.date())
    if isinstance(date_val, str) and date_val:
        return date_val
    return ""


def build_rag_context(
    rag_query: Optional[str], content: str, num_results: int = 3
) -> str:
    """
    Build retrieval context using DPR over the news-please dataset.
    - If rag_query is empty, fall back to the leading slice of content.
    """
    query = (rag_query or "").strip() or str(content or "")[:200].strip()
    if not query:
        return ""

    from .retrieval_dpr import get_dpr

    dpr = get_dpr()
    scores, retrieved_examples = dpr.search(query, k=num_results)

    parts = ["Real news writing reference (for realism boundary only; not factual grounding):\n"]
    cnt = 0
    for score, rex in zip(scores, retrieved_examples):
        desc = rex.get("description", "")
        if not desc:
            continue
        parts.append(f"{desc}\n")
        cnt += 1
        if cnt >= num_results:
            break

    return "\n".join(parts).strip()

class APIEngine:
    """使用 OpenAI API（或相容 API）生成文本"""

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it via environment variable:\n"
                "  export OPENAI_API_KEY=sk-xxx"
            )

        print(f"[API Engine] Using {self.model} via {self.base_url}")

    def _build_prompts(
        self, real_news: dict, context: str, feedback_prompt: str = None
    ) -> tuple:
        """構建 system 和 user prompt"""
        content = real_news["text"][:1500]  # API 可以處理更多 token

        # 根據是否有 feedback 調整 system prompt
        if feedback_prompt:
            system_prompt = (
                "You are a sophisticated writer engaged in an adversarial game against a fake news detector. "
                "Your previous attempt was DETECTED as fake or too similar to the original article. "
                "You must learn from the feedback and adapt your strategy. "
                + FACT_CHANGE_INSTRUCTION
                + STYLE_CONSTRAINT
                + LENGTH_CONSTRAINT
            )
        else:
            system_prompt = (
                "You are a sophisticated writer. Your task is to rewrite a real news story "
                "to introduce believable factual errors or alter key entities (names, locations, events) "
                "while maintaining the EXACT same journalistic tone and formatting style as the original. "
                + FACT_CHANGE_INSTRUCTION
                + STYLE_CONSTRAINT
                + LENGTH_CONSTRAINT
            )

        # 構建 feedback section
        feedback_section = ""
        if feedback_prompt:
            feedback_section = f"""


CRITICAL: DISCRIMINATOR FEEDBACK - YOU MUST ADDRESS THIS

{feedback_prompt}

YOUR ADAPTATION STRATEGY:
1. First, identify which flagged words/phrases you used before
2. Replace them with neutral, professional alternatives  
3. Use the RAG context to understand what kinds of changes are plausible in real news reporting. It defines a realism boundary, not facts to copy. Do NOT ensure factual consistency with it.
4. Maintain a calm, objective journalistic voice throughout
5. DO NOT use words like "shocking", "unbelievable", "sources say", etc.
"""

        user_prompt = f"""
Original Real News:
{content}{feedback_section}

Task:
Rewrite the news above to be fake but realistic.

REQUIRED FACTUAL EDITS:
- Introduce 1-2 factual modifications that change the meaning of the story.
- At least ONE of the following must be changed:
* the main person or organization involved,
* the location,
* the time/date or time period,
* key numbers (amounts, years, percentages, counts),
* the cause or the outcome of the events.
- The modified story must remain coherent and plausible.
- You MUST NOT merely paraphrase sentences or replace words with synonyms while keeping all facts the same.

Retrieved Writing Reference (for realism boundary; do NOT copy specific facts):
{context}

How to use the reference:
- Use the retrieved articles as a realism boundary, not as facts to copy.
- Observe what kinds of details are typically reported for similar events (who speaks, where announcements happen, what numbers look reasonable).
- When changing facts, keep them within ranges and patterns commonly seen in real news reporting.
- Do NOT copy specific facts, names, or sentences from the reference.
- Do NOT introduce a new unrelated event.
- If the reference conflicts with the original story, keep your story internally consistent.
- Do NOT make factual changes that would look unusual or implausible compared to how similar real news events are typically reported.

CRITICAL FORMATTING RULES:
1. Do NOT use any Markdown formatting (no ###, no **, no bullet points).
2. Do NOT add extra blank lines between paragraphs.
3. Keep the EXACT same paragraph structure as the original.
4. Keep the overall length similar to the original article (stay roughly within ±20% of the original length).
5. Do NOT add long background sections or speculative analysis that are not implied by the original article.
6. Start directly with the news content (e.g., "(CNN)..." or similar).
7. Output ONLY the rewritten fake news article, nothing else.
8. DO NOT include the original text, headings, labels (e.g., "Modified version:"), or any explanations—only the final rewritten article content.
""".strip()

        return system_prompt, user_prompt

    def generate_fake_news(
        self,
        real_news: dict,
        context: str,
        feedback_prompt: str = None,
        train_mode: bool = False,
    ) -> Union[str, Dict[str, object]]:
        """呼叫 API 生成假新聞"""
        system_prompt, user_prompt = self._build_prompts(
            real_news, context, feedback_prompt
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_tokens = _compute_max_new_tokens(real_news["text"], tokenizer=None)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.5,
            "top_p": 0.8,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            generated_text = data["choices"][0]["message"]["content"].strip()
            if not train_mode:
                return generated_text
            return {
                "text": generated_text,
                "prompt": "",
                "prompt_ids": None,
                "generated_ids": None,
            }

        except requests.exceptions.RequestException as e:
            print(f"[API Error] {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"[API Response] {e.response.text[:500]}")
            return f"[Generation failed: {str(e)[:100]}]"


# ==========================================
# 4. 本地引擎 (Transformers)
# ==========================================


class LocalEngine:
    """使用本地 HuggingFace 模型生成文本（需要 GPU）"""

    def __init__(self, model_id: str = None):
        # 延遲導入，只有需要時才載入 torch
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = model_id or LOCAL_MODEL_ID
        print(f"[Local Engine] Loading model: {model_id} (fp16)")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.use_lora = GEN_USE_LORA
        if self.use_lora:
            lora_cfg = LoraConfig(
                r=GEN_LORA_R,
                lora_alpha=GEN_LORA_ALPHA,
                lora_dropout=GEN_LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
            )
            base = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else None,
            ).to(self.device)
            self.model = get_peft_model(base, lora_cfg)
            # 便於下游日誌顯示
            setattr(self.model, "lora_r", GEN_LORA_R)
            setattr(self.model, "lora_alpha", GEN_LORA_ALPHA)
            setattr(self.model, "lora_dropout", GEN_LORA_DROPOUT)
            self.base_model = None  # reuse same weights; disable adapters when needed for KL
            print(
                f"[Local Engine] LoRA enabled (r={GEN_LORA_R}, alpha={GEN_LORA_ALPHA}, dropout={GEN_LORA_DROPOUT})"
            )
            if hasattr(self.model, "print_trainable_parameters"):
                self.model.print_trainable_parameters()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16 if self.device.type == "cuda" else None
            ).to(self.device)
            self.base_model = None

        self.model.eval()

    def _build_prompts(
        self, real_news: dict, context: str, feedback_prompt: str = None
    ) -> tuple:
        """構建 system 和 user prompt（與 APIEngine 相同邏輯）"""
        content = real_news["text"][:1000]

        if feedback_prompt:
            system_prompt = (
                "You are a sophisticated writer engaged in an adversarial game against a fake news detector. "
                "Your previous attempt was DETECTED as fake or too similar to the original article. "
                "You must learn from the feedback and adapt your strategy. "
                + FACT_CHANGE_INSTRUCTION
                + STYLE_CONSTRAINT
                + LENGTH_CONSTRAINT
            )
        else:
            system_prompt = (
                "You are a sophisticated writer. Your task is to rewrite a real news story "
                "to introduce believable factual errors or alter key entities (names, locations, events) "
                "while maintaining the EXACT same journalistic tone and formatting style as the original. "
                + FACT_CHANGE_INSTRUCTION
                + STYLE_CONSTRAINT
                + LENGTH_CONSTRAINT
            )

        feedback_section = ""
        if feedback_prompt:
            feedback_section = f"""


CRITICAL: DISCRIMINATOR FEEDBACK - YOU MUST ADDRESS THIS
{feedback_prompt}

YOUR ADAPTATION STRATEGY:
1. First, identify which flagged words/phrases you used before
2. Replace them with neutral, professional alternatives  
3. Use the RAG context to understand what kinds of changes are plausible in real news reporting. It defines a realism boundary, not facts to copy. Do NOT ensure factual consistency with it.
4. Maintain a calm, objective journalistic voice throughout
5. DO NOT use words like "shocking", "unbelievable", "sources say", etc.
"""

        user_prompt = f"""
Original Real News:
{content}{feedback_section}

Task:
Rewrite the news above to be fake but realistic.

REQUIRED FACTUAL EDITS:
- Introduce 1–2 factual modifications that change the meaning of the story.
- At least ONE of the following must be changed:
* the main person or organization involved,
* the location,
* the time/date or time period,
* key numbers (amounts, years, percentages, counts),
* the cause or the outcome of the events.
- The modified story must remain coherent and plausible.
- You MUST NOT merely paraphrase sentences or replace words with synonyms while keeping all facts the same.

Retrieved Writing Reference (for realism boundary; do NOT copy specific facts):
{context}


How to use the reference:
- Use the retrieved articles as a realism boundary, not as facts to copy.
- Observe what kinds of details are typically reported for similar events (who speaks, where announcements happen, what numbers look reasonable).
- When changing facts, keep them within ranges and patterns commonly seen in real news reporting.
- Do NOT copy specific facts, names, or sentences from the reference.
- Do NOT introduce a new unrelated event.
- If the reference conflicts with the original story, keep your story internally consistent.
- Do NOT make factual changes that would look unusual or implausible compared to how similar real news events are typically reported.

CRITICAL FORMATTING RULES:
1. Do NOT use any Markdown formatting (no ###, no **, no bullet points).
2. Do NOT add extra blank lines between paragraphs.
3. Keep the EXACT same paragraph structure as the original.
4. Keep the overall length similar to the original article (stay roughly within ±20% of the original length).
5. Do NOT add long background sections or speculative analysis that are not implied by the original article.
6. Start directly with the news content (e.g., "(CNN)..." or similar).
7. Output ONLY the rewritten fake news article, nothing else.
8. DO NOT include the original text, headings, labels (e.g., "Modified version:"), or any explanations—only the final rewritten article content.
""".strip()

        return system_prompt, user_prompt

    def generate_fake_news(
        self,
        real_news: dict,
        context: str,
        feedback_prompt: str = None,
        train_mode: bool = False,
    ) -> Union[str, Dict[str, object]]:
        """使用本地模型生成假新聞；train_mode 時會回傳 token 資訊"""
        import torch

        system_prompt, user_prompt = self._build_prompts(
            real_news, context, feedback_prompt
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        
        # 💡 根據原文長度動態決定 max_new_tokens，避免生成過長
        dyn_max_new_tokens = _compute_max_new_tokens(
            real_news["text"], tokenizer=self.tokenizer
        )

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **enc,
                max_new_tokens=dyn_max_new_tokens,
                temperature=0.5,
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if was_training:
            self.model.train()

        prompt_len = enc["input_ids"].shape[1]
        generated_ids = generated[0][prompt_len:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        if not train_mode:
            return generated_text

        return {
            "text": generated_text,
            "prompt": prompt,
            "prompt_ids": enc["input_ids"][0].detach().cpu(),
            "generated_ids": generated_ids.detach().cpu(),
        }


# ==========================================
# 5. 統一介面: FakeNewsGenerator
# ==========================================


class FakeNewsGenerator:
    """統一的假新聞生成器，自動選擇 API 或本地模式"""

    def __init__(self, model_id: str = None, mode: str = None, api_key: str = None):
        self.mode = mode or GEN_MODE

        if self.mode == "api":
            # API 模式：忽略傳入的 model_id（那是本地模型名），使用環境變數 OPENAI_MODEL
            # 只有當 model_id 看起來像 OpenAI 模型時才使用
            api_model = None
            if model_id and model_id.startswith(("gpt-", "o1-", "chatgpt-")):
                api_model = model_id
            self.engine = APIEngine(api_key=api_key, model=api_model)
        elif self.mode == "local":
            self.engine = LocalEngine(model_id=model_id)
        else:
            raise ValueError(f"Unknown GEN_MODE: {self.mode}. Use 'api' or 'local'.")

        print(f"[FakeNewsGenerator] Initialized in '{self.mode}' mode")

    def get_trainable_components(self):
        """Return (model, tokenizer, base_model) for local training."""
        if self.mode != "local":
            raise ValueError("Trainable components only available in local mode.")
        return (
            self.engine.model,
            self.engine.tokenizer,
            getattr(self.engine, "base_model", None),
        )

    def generate(
        self,
        content: str,
        feedback_prompt: str = None,
        use_rag: bool = True,
        context_override: str = None,
        rag_query: str = None,
        num_rag_results: int = 3,
        lang: str = "en",
        train_mode: bool = False,
    ) -> Union[str, Dict[str, object]]:
        """生成假新聞"""
        real_news = {"text": content}

        # RAG 處理：只有非空 override 才覆蓋，否則用 DPR 取 top-k 內文 context
        override = (context_override or "").strip()
        if use_rag:
            if override:
                rag_context = override
            else:
                rag_context = build_rag_context(
                    rag_query=rag_query,
                    content=content,
                    num_results=num_rag_results,
                )
        else:
            rag_context = ""

        fake_news = self.engine.generate_fake_news(
            real_news, rag_context, feedback_prompt, train_mode=train_mode
        )
        return fake_news


# ==========================================
# 6. 便捷函數（向後相容）
# ==========================================

_global_generator = None


def generator(
    content: str,
    feedback_prompt: str = None,
    use_rag: bool = True,
    rag_query: str = None,
    num_rag_results: int = 3,
    lang: str = "en",
    context_override: str = None,
    model_id: str = None,
) -> Union[str, Dict[str, object]]:
    """便捷函數，自動管理全局 generator 實例"""
    global _global_generator
    if _global_generator is None:
        _global_generator = FakeNewsGenerator(model_id=model_id)

    return _global_generator.generate(
        content,
        feedback_prompt,
        use_rag,
        context_override,
        rag_query,
        num_rag_results,
        lang,
        train_mode=False,
    )


def reset_generator():
    """Reset the shared generator instance so new settings take effect."""
    global _global_generator
    _global_generator = None


# ==========================================
# 7. Demo / 測試
# ==========================================


def main():
    """簡單測試 - 生成一則假新聞"""
    print("=" * 60)
    print("RAG²AN Generator Demo")
    print(f"Mode: {GEN_MODE}")
    print("=" * 60)

    # 測試新聞
    test_news = {
        "text": """(CNN) Apple announced today that it will release a new iPhone model next month. 
The company's CEO Tim Cook revealed the news at a press conference in Cupertino, California.
The new device is expected to feature improved battery life and a faster processor.
Industry analysts predict strong sales for the holiday season.""",
    }

    print("\n[Original News]")
    print(test_news["text"])

    print("\n[Generating fake version...]")
    gen = FakeNewsGenerator()
    fake = gen.generate(
        content=test_news["text"],
        use_rag=True,
    )

    print("\n[Generated Fake News]")
    print(fake)

    # 測試有 feedback 的情況
    print("\n" + "=" * 60)
    print("[Testing with feedback...]")
    print("=" * 60)

    feedback = """
=== DISCRIMINATOR FEEDBACK (Round 1) ===
Detection Result: Your previous version was classified as FAKE
Confidence: HIGH (85% fake probability)

Flagged Suspicious Terms: shocking(0.89), sources(0.76), reportedly(0.71)

Improvement Instructions:
  1. Replace sensationalist words with neutral alternatives
  2. Use specific, named sources instead of vague attributions
"""

    fake_v2 = gen.generate(
        content=test_news["text"],
        feedback_prompt=feedback,
        use_rag=True,
    )

    print("\n[Generated Fake News v2 (with feedback)]")
    print(fake_v2)


if __name__ == "__main__":
    main()
