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
from typing import Optional

# ==========================================
# 1. 配置設定 (Configuration)
# ==========================================

# 模式選擇：'api', 'local', 或 'lora'
GEN_MODE = os.environ.get("GEN_MODE", "api")

# API 設定（OpenAI 相容）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# 本地模型設定
LOCAL_MODEL_ID = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# LoRA 設定
LORA_PATH = os.environ.get("LORA_PATH", None)  # LoRA checkpoint 路徑
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"  # 是否使用 4-bit 量化

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
                "CRITICAL FORMAT: You MUST output in 'Title:' and 'Body:' format. "
                "The Title should be a single line. The Body should follow on a new line after 'Body:'."
            )
        else:
            system_prompt = (
                "You are a professional news writer. Your task is to rewrite a real news story "
                "by subtly altering key facts, names, locations, or numbers. "
                "CRITICAL: You MUST mimic the EXACT writing style, tone, vocabulary level, and sentence structure of the original article. "
                "Use the same formal, objective, journalistic language. Avoid sensational words like 'shocking', 'unbelievable', 'bombshell'. "
                "Avoid vague attributions like 'sources say', 'reportedly', 'allegedly'. Instead, use concrete sources like the original. "
                "The rewritten article should read like it came from the same professional news outlet as the original. "
                "CRITICAL FORMAT: You MUST output in 'Title:' and 'Body:' format. "
                "The Title should be a single line. The Body should follow on a new line after 'Body:'."
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
Rewrite the news above by altering some key facts (e.g., change numbers, names, locations, or specific details). 

STYLE REQUIREMENTS (CRITICAL):
- Use the EXACT same writing style, tone, and vocabulary as the original
- Match the sentence structure and paragraph flow of the original  
- Use formal, objective language like a professional journalist
- Avoid sensational words (shocking, unbelievable, bombshell, stunning, etc.)
- Use specific named sources, not vague ones (no "sources say", "reportedly", "allegedly")
- Make the altered facts blend naturally into the professional tone

FORMATTING RULES:
1. Output in this EXACT format:
   Title: [Your rewritten title here]
   Body: [Your rewritten article here]
2. Do NOT use Markdown formatting (no ###, no **, no bullet points)
3. Keep the same paragraph structure as the original
4. The Title should be a single line
5. The Body should start on a new line after "Body:"
        """.strip()

        return system_prompt, user_prompt

    def generate_fake_news(
        self, real_news: dict, context: str, feedback_prompt: str = None
    ) -> str:
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
            generated_text = data["choices"][0]["message"]["content"]
            return generated_text.strip()

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
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

        model_id = model_id or LOCAL_MODEL_ID
        print(f"[Local Engine] Loading model: {model_id} (fp16)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )

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
                "CRITICAL FORMAT: You MUST output in 'Title:' and 'Body:' format. "
                "The Title should be a single line. The Body should follow on a new line after 'Body:'."
            )
        else:
            system_prompt = (
                "You are a professional news writer. Your task is to rewrite a real news story "
                "by subtly altering key facts, names, locations, or numbers. "
                "CRITICAL: You MUST mimic the EXACT writing style, tone, vocabulary level, and sentence structure of the original article. "
                "Use the same formal, objective, journalistic language. Avoid sensational words like 'shocking', 'unbelievable', 'bombshell'. "
                "Avoid vague attributions like 'sources say', 'reportedly', 'allegedly'. Instead, use concrete sources like the original. "
                "The rewritten article should read like it came from the same professional news outlet as the original. "
                "CRITICAL FORMAT: You MUST output in 'Title:' and 'Body:' format. "
                "The Title should be a single line. The Body should follow on a new line after 'Body:'."
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
Rewrite the news above by altering some key facts (e.g., change numbers, names, locations, or specific details). 

STYLE REQUIREMENTS (CRITICAL):
- Use the EXACT same writing style, tone, and vocabulary as the original
- Match the sentence structure and paragraph flow of the original  
- Use formal, objective language like a professional journalist
- Avoid sensational words (shocking, unbelievable, bombshell, stunning, etc.)
- Use specific named sources, not vague ones (no "sources say", "reportedly", "allegedly")
- Make the altered facts blend naturally into the professional tone

FORMATTING RULES:
1. Output in this EXACT format:
   Title: [Your rewritten title here]
   Body: [Your rewritten article here]
2. DO NOT use Markdown formatting (no ###, no **, no bullet points)
3. Keep the same paragraph structure as the original
4. The Title should be a single line
5. The Body should start on a new line after "Body:"
        """.strip()

        return system_prompt, user_prompt

    def generate_fake_news(
        self, real_news: dict, context: str, feedback_prompt: str = None
    ) -> str:
        """使用本地模型生成假新聞"""
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
        outputs = self.pipe(prompt)
        generated_text = outputs[0].get("generated_text", "")

        # 清理輸出
        if (
            isinstance(generated_text, str)
            and prompt
            and generated_text.startswith(prompt)
        ):
            generated_text = generated_text[len(prompt) :]

        split_tok = "<|start_header_id|>assistant<|end_header_id|>"
        if split_tok in generated_text:
            generated_text = generated_text.split(split_tok)[-1].strip()

        return generated_text


# ==========================================
# 4.5. LoRA 引擎 (Transformers + LoRA)
# ==========================================


class LoRALocalEngine:
    """使用 LoRA fine-tuned 模型生成文本（需要 GPU）"""

    def __init__(self, model_id: str = None, lora_path: str = None, use_4bit: bool = True):
        # 延遲導入
        import torch
        from transformers import (
            pipeline,
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        model_id = model_id or LOCAL_MODEL_ID
        lora_path = lora_path or LORA_PATH
        
        print(f"[LoRA Engine] Loading model: {model_id}")
        if lora_path:
            print(f"[LoRA Engine] With LoRA weights: {lora_path}")
        if use_4bit:
            print(f"[LoRA Engine] Using 4-bit quantization")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 根據是否使用 4-bit 量化來載入模型
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="auto"
            )

        # 載入 LoRA weights（如果有）
        if lora_path and os.path.exists(lora_path):
            try:
                from peft import PeftModel
                print(f"[LoRA Engine] Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                print(f"[LoRA Engine] LoRA adapter loaded successfully")
            except ImportError:
                print(f"[LoRA Engine] Warning: peft not installed, skipping LoRA")
            except Exception as e:
                print(f"[LoRA Engine] Warning: Failed to load LoRA: {e}")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )

    def _build_prompts(
        self, real_news: dict, context: str, feedback_prompt: str = None
    ) -> tuple:
        """構建 system 和 user prompt（與 LocalEngine 相同邏輯）"""
        # 重用 LocalEngine 的邏輯
        from generator import LocalEngine
        temp_engine = LocalEngine.__new__(LocalEngine)
        return temp_engine._build_prompts(real_news, context, feedback_prompt)

    def generate_fake_news(
        self, real_news: dict, context: str, feedback_prompt: str = None
    ) -> str:
        """使用 LoRA 模型生成假新聞"""
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
        outputs = self.pipe(prompt)
        generated_text = outputs[0].get("generated_text", "")

        # 清理輸出
        if (
            isinstance(generated_text, str)
            and prompt
            and generated_text.startswith(prompt)
        ):
            generated_text = generated_text[len(prompt) :]

        split_tok = "<|start_header_id|>assistant<|end_header_id|>"
        if split_tok in generated_text:
            generated_text = generated_text.split(split_tok)[-1].strip()

        return generated_text


# ==========================================
# 5. 統一介面: FakeNewsGenerator
# ==========================================


class FakeNewsGenerator:
    """統一的假新聞生成器，自動選擇 API、本地或 LoRA 模式"""

    def __init__(
        self,
        model_id: str = None,
        mode: str = None,
        api_key: str = None,
        lora_path: str = None,
        use_4bit: bool = None,
    ):
        self.mode = mode or GEN_MODE
        lora_path = lora_path or LORA_PATH
        use_4bit = use_4bit if use_4bit is not None else USE_4BIT

        if self.mode == "api":
            # API 模式：忽略傳入的 model_id（那是本地模型名），使用環境變數 OPENAI_MODEL
            # 只有當 model_id 看起來像 OpenAI 模型時才使用
            api_model = None
            if model_id and model_id.startswith(("gpt-", "o1-", "chatgpt-")):
                api_model = model_id
            self.engine = APIEngine(api_key=api_key, model=api_model)
        elif self.mode == "lora":
            self.engine = LoRALocalEngine(
                model_id=model_id, lora_path=lora_path, use_4bit=use_4bit
            )
        elif self.mode == "local":
            self.engine = LocalEngine(model_id=model_id)
        else:
            raise ValueError(
                f"Unknown GEN_MODE: {self.mode}. Use 'api', 'local', or 'lora'."
            )

        print(f"[FakeNewsGenerator] Initialized in '{self.mode}' mode")

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
    ) -> str:
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
            real_news, rag_context, feedback_prompt
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
) -> str:
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
