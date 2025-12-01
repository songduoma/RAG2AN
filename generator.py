from datasets import load_dataset
import torch
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

"""
更新說明：
1. main() 函式維持讀取前 5 筆資料 (test[:5])。
2. 修改：最終輸出時顯示「完整真新聞全文 (Full Text)」而非摘要，方便進行比較。
"""

# ==========================================
# 1. 配置設定 (Configuration)
# ==========================================

# 使用 Llama 3.1-8B Instruct 或 Qwen (需 Hugging Face 權限或公開模型)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ==========================================
# 2. 工具函數: Wikipedia RAG
# ==========================================

def search_wikipedia(query: str, num_results: int = 3, lang: str = "en", verbose: bool = False) -> str:
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
                r2 = requests.get(search_url, params=detail_params, headers=headers, timeout=10)
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
# 3. 工具類別: Llama 引擎
# ==========================================

class LlamaEngine:
    def __init__(self, model_id: str):
        print(f"Loading model: {model_id} (fp16)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )

    def generate_fake_news(self, real_news, context: str, feedback_prompt: str = None) -> str:
        title = real_news["title"]
        content = real_news["text"][:1000] # 輸入給模型的 prompt 仍需適當截斷以免爆顯存

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
            feedback_section = f"\n\nFeedback (AVOID these words/styles):\n{feedback_prompt}\nPlease follow this feedback strictly."

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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt)
        generated_text = outputs[0].get("generated_text", "")

        if isinstance(generated_text, str) and prompt and generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        split_tok = "<|start_header_id|>assistant<|end_header_id|>"
        if split_tok in generated_text:
            generated_text = generated_text.split(split_tok)[-1].strip()

        return generated_text

# ==========================================
# 4. Generator 函式（供外部呼叫）
# ==========================================

class FakeNewsGenerator:
    def __init__(self, model_id: str = MODEL_ID):
        self.llama = LlamaEngine(model_id)
    
    def generate(self, title: str, content: str, feedback_prompt: str = None, use_rag: bool = True, 
                 context_override: str = None, rag_query: str = None, num_rag_results: int = 3, lang: str = "en") -> str:
        
        real_news = {"title": title, "text": content}
        
        rag_context = context_override if context_override is not None else ""
        if use_rag and context_override is None:
            query = rag_query if rag_query else content[:50].replace("\n", " ")
            rag_context = search_wikipedia(query, num_results=num_rag_results, lang=lang)
        
        fake_news = self.llama.generate_fake_news(real_news, rag_context, feedback_prompt)
        return fake_news

_global_generator = None

def generator(title: str, content: str, feedback_prompt: str = None, use_rag: bool = True,
              rag_query: str = None, num_rag_results: int = 3, lang: str = "en",
              context_override: str = None, model_id: str = None) -> str:
    global _global_generator
    if _global_generator is None:
        _global_generator = FakeNewsGenerator(model_id if model_id else MODEL_ID)
    
    return _global_generator.generate(title, content, feedback_prompt, use_rag, context_override, rag_query, num_rag_results, lang)

# ==========================================
# 5. 主程序（修改版）
# ==========================================

def main():
    # --- A. 載入前 5 筆資料 ---
    print("Loading Dataset (CNN/DailyMail, test[:5])...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:5]")

    # --- B. 初始化 Llama 引擎 ---
    llama = LlamaEngine(MODEL_ID)

    comparison_records = []

    # --- C. 迴圈處理 5 筆新聞 ---
    for i, news_item in enumerate(dataset):
        print(f"\n>>> Processing News {i+1} / 5 ...")

        article_text = news_item["article"]
        formatted_news = {
            "title": "Breaking News",
            "text": article_text
        }

        # --- D. RAG ---
        search_query = formatted_news["text"][:50].replace("\n", " ")
        rag_context = search_wikipedia(search_query, num_results=3, lang="en")

        # --- E. 生成假新聞 ---
        fake_news = llama.generate_fake_news(formatted_news, rag_context)

        # 存入列表 (這裡修改為儲存全文)
        comparison_records.append({
            "id": i + 1,
            "real_text": article_text,  # <--- 修改處：儲存完整文章
            "fake_result": fake_news
        })

    # --- F. 最終輸出比較 ---
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT (Top 5)")
    print("="*80)

    for record in comparison_records:
        print(f"\n[Record #{record['id']}]")
        
        # 顯示完整真新聞
        print("-" * 20 + " REAL NEWS (Full Text) " + "-" * 20)
        print(record['real_text'])
        
        # 顯示假新聞
        print("\n" + "-" * 20 + " GENERATED FAKE NEWS " + "-" * 20)
        print(record['fake_result'])
        
        print("\n" + "_"*80)

if __name__ == "__main__":
    main()
