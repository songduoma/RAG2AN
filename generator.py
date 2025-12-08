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

import copy
import os
import requests
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

# API 設定（OpenAI 相容）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 本地模型設定
LOCAL_MODEL_ID = os.environ.get("GEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

# 為了向後相容，保留 MODEL_ID
MODEL_ID = LOCAL_MODEL_ID

# ==========================================
# 2. 工具函數: Wikipedia RAG
# ==========================================


def search_wikipedia(
    query: str, num_results: int = 3, lang: str = "en", verbose: bool = False
) -> str:
    """
    用 Wikipedia 官方 API 做簡單 RAG
    """
    if verbose:
        print(f"[Wiki] query = {query!r}")

    headers = {
        "User-Agent": "NTU-ADL-FinalProject/0.1",
        "Accept": "application/json",
    }

    try:
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
        }

        r = requests.get(search_url, params=search_params, headers=headers, timeout=10)

        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return ""

        data = r.json()
        if "query" not in data or "search" not in data["query"]:
            return ""

        context_lines = []
        for item in data["query"]["search"]:
            title = item.get("title", "")
            pageid = item.get("pageid")

            extract = ""
            if pageid is not None:
                detail_params = {
                    "action": "query",
                    "prop": "extracts",
                    "pageids": pageid,
                    "exintro": True,
                    "explaintext": True,
                    "format": "json",
                }
                r2 = requests.get(
                    search_url, params=detail_params, headers=headers, timeout=10
                )
                d2 = r2.json()
                pages = d2.get("query", {}).get("pages", {})
                page = pages.get(str(pageid), {})
                extract = page.get("extract", "")

            line = f"- Title: {title}\n  Snippet: {extract[:300]}"
            context_lines.append(line)

        context_text = "\n".join(context_lines)
        return context_text

    except Exception as e:
        print(f"[Wiki] Error: {e}")
        return f"- Title: Simulation\n  Snippet: Could not fetch Wikipedia for {query[:20]}."


# ==========================================
# 3. API 引擎 (OpenAI 相容)
# ==========================================


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
                "Your previous attempt was DETECTED as fake. You must learn from the feedback and adapt your strategy. "
                "Your goal: rewrite the news to introduce subtle factual errors while EVADING detection. "
                "KEY STRATEGY: The detector flagged specific words and patterns. You MUST avoid these tells. "
                "Write in a neutral, professional journalistic tone. Avoid sensationalism. "
                "IMPORTANT: Do NOT use Markdown formatting (no ###, no **, no bullet points). "
                "Keep the same paragraph structure and spacing as the original news article."
            )
        else:
            system_prompt = (
                "You are a sophisticated writer. Your task is to rewrite a real news story "
                "to introduce believable factual errors or alter key entities (names, locations, events) "
                "while maintaining the EXACT same journalistic tone and formatting style as the original. "
                "The goal is to create a piece of 'Fake News' that is plausible enough to fool fact-checkers. "
                "IMPORTANT: Do NOT use Markdown formatting (no ###, no **, no bullet points). "
                "Keep the same paragraph structure and spacing as the original news article."
            )

        # 構建 feedback section
        feedback_section = ""
        if feedback_prompt:
            feedback_section = f"""

╔══════════════════════════════════════════════════════════════╗
║  CRITICAL: DISCRIMINATOR FEEDBACK - YOU MUST ADDRESS THIS    ║
╚══════════════════════════════════════════════════════════════╝

{feedback_prompt}

YOUR ADAPTATION STRATEGY:
1. First, identify which flagged words/phrases you used before
2. Replace them with neutral, professional alternatives  
3. Ensure your facts are internally consistent with the RAG context
4. Maintain a calm, objective journalistic voice throughout
5. DO NOT use words like "shocking", "unbelievable", "sources say", etc.
"""

        user_prompt = f"""
Background Information (RAG Context):
{context}

Original Real News:
{content}{feedback_section}

Task:
Rewrite the news above to be fake but realistic. 
CRITICAL FORMATTING RULES:
1. Do NOT use any Markdown formatting (no ###, no **, no bullet points)
2. Do NOT add extra blank lines between paragraphs
3. Keep the EXACT same paragraph structure as the original
4. Start directly with the news content (e.g., "(CNN)..." or similar)
5. Output ONLY the rewritten fake news article, nothing else
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

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
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

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device.type == "cuda" else None
        ).to(self.device)

        self.use_lora = GEN_USE_LORA
        if self.use_lora:
            lora_cfg = LoraConfig(
                r=GEN_LORA_R,
                lora_alpha=GEN_LORA_ALPHA,
                lora_dropout=GEN_LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base_model, lora_cfg)
            # 便於下游日誌顯示
            setattr(self.model, "lora_r", GEN_LORA_R)
            setattr(self.model, "lora_alpha", GEN_LORA_ALPHA)
            setattr(self.model, "lora_dropout", GEN_LORA_DROPOUT)
            self.base_model = copy.deepcopy(base_model).eval()
            for p in self.base_model.parameters():
                p.requires_grad_(False)
            print(
                f"[Local Engine] LoRA enabled (r={GEN_LORA_R}, alpha={GEN_LORA_ALPHA}, dropout={GEN_LORA_DROPOUT})"
            )
            if hasattr(self.model, "print_trainable_parameters"):
                self.model.print_trainable_parameters()
        else:
            self.model = base_model
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
                "Your previous attempt was DETECTED as fake. You must learn from the feedback and adapt your strategy. "
                "Your goal: rewrite the news to introduce subtle factual errors while EVADING detection. "
                "KEY STRATEGY: The detector flagged specific words and patterns. You MUST avoid these tells. "
                "Write in a neutral, professional journalistic tone. Avoid sensationalism. "
                "IMPORTANT: Do NOT use Markdown formatting (no ###, no **, no bullet points). "
                "Keep the same paragraph structure and spacing as the original news article."
            )
        else:
            system_prompt = (
                "You are a sophisticated writer. Your task is to rewrite a real news story "
                "to introduce believable factual errors or alter key entities (names, locations, events) "
                "while maintaining the EXACT same journalistic tone and formatting style as the original. "
                "The goal is to create a piece of 'Fake News' that is plausible enough to fool fact-checkers. "
                "IMPORTANT: Do NOT use Markdown formatting (no ###, no **, no bullet points). "
                "Keep the same paragraph structure and spacing as the original news article."
            )

        feedback_section = ""
        if feedback_prompt:
            feedback_section = f"""

╔══════════════════════════════════════════════════════════════╗
║  CRITICAL: DISCRIMINATOR FEEDBACK - YOU MUST ADDRESS THIS    ║
╚══════════════════════════════════════════════════════════════╝

{feedback_prompt}

YOUR ADAPTATION STRATEGY:
1. First, identify which flagged words/phrases you used before
2. Replace them with neutral, professional alternatives  
3. Ensure your facts are internally consistent with the RAG context
4. Maintain a calm, objective journalistic voice throughout
5. DO NOT use words like "shocking", "unbelievable", "sources say", etc.
"""

        user_prompt = f"""
Background Information (RAG Context):
{context}

Original Real News:
{content}{feedback_section}

Task:
Rewrite the news above to be fake but realistic. 
CRITICAL FORMATTING RULES:
1. Do NOT use any Markdown formatting (no ###, no **, no bullet points)
2. Do NOT add extra blank lines between paragraphs
3. Keep the EXACT same paragraph structure as the original
4. Start directly with the news content (e.g., "(CNN)..." or similar)
5. Output ONLY the rewritten fake news article, nothing else
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

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                **enc,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
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
        title: str,
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
        real_news = {"title": title, "text": content}

        # RAG 處理
        rag_context = context_override if context_override is not None else ""
        if use_rag and context_override is None:
            query = rag_query if rag_query else content[:50].replace("\n", " ")
            rag_context = search_wikipedia(
                query, num_results=num_rag_results, lang=lang
            )

        fake_news = self.engine.generate_fake_news(
            real_news, rag_context, feedback_prompt, train_mode=train_mode
        )
        return fake_news


# ==========================================
# 6. 便捷函數（向後相容）
# ==========================================

_global_generator = None


def generator(
    title: str,
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
        title,
        content,
        feedback_prompt,
        use_rag,
        context_override,
        rag_query,
        num_rag_results,
        lang,
        train_mode=False,
    )


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
        "title": "Tech Company Announces New Product",
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
        title=test_news["title"],
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
        title=test_news["title"],
        content=test_news["text"],
        feedback_prompt=feedback,
        use_rag=True,
    )

    print("\n[Generated Fake News v2 (with feedback)]")
    print(fake_v2)


if __name__ == "__main__":
    main()
