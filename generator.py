from datasets import load_dataset
import torch
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

"""
1. 把功能包裝成 generator 函式，方便外部呼叫
2. 在 LlamaEngine.generate_fake_news 中增加 feedback_prompt 參數，讓使用者可以提供 feedback 來改善生成品質


from rag_llama_demo_en import generator

//import
from class_generator import FakeNewsGenerator
my_gen = FakeNewsGenerator() 

//input
title = "台積電宣布在南極設廠" 
content = "台積電今日宣布重大計畫，將與企鵝合作開發低溫超導晶片..."
feedback = "避免使用 '企鵝'，改用 '當地專家'，語氣要更嚴肅一點"

//call class
fake_news = my_gen.generate( 
title=title, 
content=content, 
feedback_prompt=feedback,
use_rag=True,  # 是否要去維基百科查資料增加可信度 
lang="zh"  # 設定維基百科搜尋語言為中文
)

//output
print(fake_news)


"""






# ==========================================
# 1. 配置設定 (Configuration)
# ==========================================

# 使用 Llama 3.1-8B Instruct（需 Hugging Face 權限）
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# 如果測試跑不動，也可以暫時改成小一點的模型：
# MODEL_ID = "gpt2"

# ==========================================
# 2. 工具函數: Wikipedia RAG
# ==========================================


def search_wikipedia(query: str, num_results: int = 3, lang: str = "en", verbose: bool = False) -> str:
    """
    用 Wikipedia 官方 API 做簡單 RAG：
    1) 先用 search API 找到相關條目
    2) 再用 pageid 抓每個條目的摘要（extract）
    3) 回傳整理好的文字，給 Llama 當 Background Context
    """
    if verbose:
        print(f"[Wiki] query = {query!r}")

    # 官方建議要帶 user-agent，避免被擋
    headers = {
        "User-Agent": "NTU-ADL-FinalProject/0.1",
        "Accept": "application/json",
    }

    try:
        # Step 1: search
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
        }

        r = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        if verbose:
            print(f"[Wiki] HTTP status = {r.status_code}")
            print(f"[Wiki] raw text (前80字) = {r.text[:80]!r}")

        # 如果不是 200，或看起來不像 JSON，就直接 fallback
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            raise RuntimeError(f"Unexpected response from Wikipedia: status={r.status_code}")

        data = r.json()

        if "query" not in data or "search" not in data["query"]:
            print("[Wiki] No search results in JSON.")
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
                    "exintro": True,        # 只要開頭
                    "explaintext": True,    # 純文字
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
        # 🔁 fallback：不要讓整個 pipeline 掛掉，給一段假的背景
        fallback = f"""
- Title: 模擬背景（Wiki 未取得正常結果）
  Snippet: 原本要從 Wikipedia 查詢「{query[:20]}」，但目前環境無法正常取得 JSON 回應。
        """.strip()
        return fallback

# ==========================================
# 3. 工具類別: Llama 3.1 生成器
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
        """
        根據真實新聞 + RAG 背景資訊，產生「假新聞版本」。
        real_news: dict, { "title": str, "text": str }
        context: str, 來自 Wikipedia 的背景摘要
        feedback_prompt: str, 可選的回饋提示，告訴 LLM 哪些字詞太假要避免使用
        """
        title = real_news["title"]
        content = real_news["text"][:1000]  # 避免太長

        system_prompt = (
            "You are a sophisticated writer. Your task is to rewrite a real news story "
            "to introduce believable factual errors or alter key entities (names, locations, events) "
            "while maintaining the journalistic tone. "
            "The goal is to create a piece of 'Fake News' that is plausible enough to fool fact-checkers. "
            "This is only for research and model training, not for real-world publishing."
        )

        # 如果有 feedback_prompt，加入額外的指示
        feedback_section = ""
        if feedback_prompt:
            feedback_section = f"""

### Feedback (Words/Phrases to AVOID - they sound too fake):
{feedback_prompt}
Please make sure NOT to use the words or phrases mentioned above, and generate more realistic content instead.
"""

        user_prompt = f"""
### Background Information (RAG Context from Wikipedia):
{context}

### Original Real News:
Title: {title}
Content: {content}{feedback_section}

### Task:
Please rewrite the news above.
1. Use the Background Information to add realistic details but twist the main facts.
2. Keep the style professional, like a news article.
3. Output format:
Title: [New Title]
Body: [New Body]
        """.strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 將對話格式轉成模型的 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(prompt)
        # pipeline 預設會回傳整個 prompt + 生成。設定 return_full_text=False 更乾淨。
        generated_text = outputs[0].get("generated_text", "")

        # 若包含前置 prompt，裁掉
        if isinstance(generated_text, str) and prompt and generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        # 如果是 Llama-3.1 的 chat 模板，可能會包含特殊 token，這裡做個簡單切割
        split_tok = "<|start_header_id|>assistant<|end_header_id|>"
        if split_tok in generated_text:
            generated_text = generated_text.split(split_tok)[-1].strip()

        return generated_text

# ==========================================
# 4. Generator 函式（供外部呼叫）
# ==========================================

class FakeNewsGenerator:
    """
    假新聞生成器類別，封裝 LlamaEngine 和 Wikipedia RAG 功能。
    可以讓外部程式方便地呼叫來產生假新聞。
    
    使用範例:
        generator = FakeNewsGenerator()
        fake_news = generator.generate(
            title="Breaking News",
            content="Original news content here...",
            feedback_prompt="Avoid using words like 'shocking', 'unbelievable'"
        )
    """
    
    def __init__(self, model_id: str = MODEL_ID):
        """
        初始化 FakeNewsGenerator。
        model_id: 使用的模型 ID，預設為 Qwen2.5-7B-Instruct
        """
        self.llama = LlamaEngine(model_id)
    
    def generate(
        self,
        title: str,
        content: str,
        feedback_prompt: str = None,
        use_rag: bool = True,
        context_override: str = None,
        rag_query: str = None,
        num_rag_results: int = 3,
        lang: str = "en"
    ) -> str:
        """
        生成假新聞的主要函式。
        
        Args:
            title: 原始新聞標題
            content: 原始新聞內容
            feedback_prompt: 回饋提示，告訴 LLM 哪些字詞太假要避免使用
                            例如: "Avoid words like 'shocking', 'unbelievable', 'sources say'"
            use_rag: 是否使用 Wikipedia RAG 來增強背景知識
            context_override: 外部提供的 RAG 背景，若提供則不再呼叫 Wikipedia
            rag_query: 自訂 RAG 查詢詞（若為 None，則使用 content 的前 50 個字）
            num_rag_results: Wikipedia 搜尋結果數量
            lang: Wikipedia 語言 ("en", "zh", "ja" 等)
        
        Returns:
            str: 生成的假新聞文本
        """
        # 準備新聞格式
        real_news = {
            "title": title,
            "text": content
        }
        
        # RAG 取得背景資訊
        rag_context = context_override if context_override is not None else ""
        if use_rag and context_override is None:
            query = rag_query if rag_query else content[:50].replace("\n", " ")
            rag_context = search_wikipedia(query, num_results=num_rag_results, lang=lang, verbose=False)
        
        # 生成假新聞
        fake_news = self.llama.generate_fake_news(real_news, rag_context, feedback_prompt)
        
        return fake_news


# 全域變數，用於延遲初始化
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
    model_id: str = None
) -> str:
    """
    假新聞生成函式（供外部直接呼叫）。
    會自動初始化模型（第一次呼叫時）。
    
    Args:
        title: 原始新聞標題
        content: 原始新聞內容
        feedback_prompt: 回饋提示，告訴 LLM 哪些字詞太假要避免使用
                        例如: "Avoid words like 'shocking', 'unbelievable'"
                        或是: "Don't use '驚爆', '震驚', '消息人士透露'"
        use_rag: 是否使用 Wikipedia RAG 來增強背景知識
        rag_query: 自訂 RAG 查詢詞（若為 None，則使用 content 的前 50 個字）
        num_rag_results: Wikipedia 搜尋結果數量
        lang: Wikipedia 語言 ("en", "zh", "ja" 等)
        model_id: 使用的模型 ID（若為 None，使用預設模型）
    
    Returns:
        str: 生成的假新聞文本
    
    使用範例:
        # 基本使用
        fake_news = generator(
            title="Stock Market Hits Record High",
            content="The stock market reached unprecedented levels today..."
        )
        
        # 使用 feedback 來改善生成品質
        fake_news = generator(
            title="Stock Market Hits Record High",
            content="The stock market reached unprecedented levels today...",
            feedback_prompt="Avoid using 'shocking discovery', 'anonymous sources', 'experts claim'"
        )
        
        # 不使用 RAG
        fake_news = generator(
            title="Local Event",
            content="A local event happened...",
            use_rag=False
        )
    """
    global _global_generator
    
    # 延遲初始化
    if _global_generator is None:
        _global_generator = FakeNewsGenerator(model_id if model_id else MODEL_ID)
    
    return _global_generator.generate(
        title=title,
        content=content,
        feedback_prompt=feedback_prompt,
        use_rag=use_rag,
        context_override=context_override,
        rag_query=rag_query,
        num_rag_results=num_rag_results,
        lang=lang
    )


# ==========================================
# 5. 主程序（Demo）
# ==========================================

def main():
    # --- A. 載入真實新聞 Dataset ---
    print("Loading Dataset (CNN/DailyMail, test[0])...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:1]")  # 只拿 1 筆

    # --- B. 初始化 Llama 引擎 ---
    llama = LlamaEngine(MODEL_ID)

    # --- C. 處理每一則新聞（這裡只有 1 則） ---
    for i, news_item in enumerate(dataset):
        print(f"\n{'='*20} Processing News {i+1} {'='*20}")

        # CNN/DailyMail 的欄位是 'article'
        article_text = news_item["article"]
        original_snippet = article_text[:300].replace("\n", " ")
        formatted_news = {
            "title": "Breaking News",  # 如果沒有 title 欄位，就先給一個 placeholder
            "text": article_text
        }

        print(f"\n[Original News Snippet]:\n{original_snippet}")

        # --- D. RAG：用 Wikipedia 當外部知識來源 ---
        search_query = formatted_news["text"][:50].replace("\n", " ")
        print(f"\n[RAG] Using Wikipedia with query:\n{search_query}")

        # lang="en" 用英文維基；之後可以改成 "ja" / "zh"
        rag_context = search_wikipedia(search_query, num_results=3, lang="en")
        print(f"\n[RAG Context from Wikipedia]:\n{rag_context}")

        # --- E. 生成假新聞（不帶 feedback） ---
        fake_news = llama.generate_fake_news(formatted_news, rag_context)
        print(f"\n[Generated Fake News (without feedback)]:\n{fake_news}")
        
        # --- F. 使用 feedback 重新生成（Demo） ---
        print(f"\n{'='*20} Regenerating with Feedback {'='*20}")
        feedback = "Avoid using words like 'shocking', 'unbelievable', 'sources say', 'breaking'"
        fake_news_v2 = llama.generate_fake_news(formatted_news, rag_context, feedback_prompt=feedback)
        print(f"\n[Generated Fake News (with feedback)]:\n{fake_news_v2}")
        
        print("-" * 80)


def demo_generator_function():
    """
    示範如何使用 generator 函式
    """
    print("=" * 60)
    print("Demo: Using generator() function")
    print("=" * 60)
    
    # 範例新聞
    sample_title = "Tech Company Announces New Product"
    sample_content = """
    A major technology company announced today the launch of their latest 
    smartphone device. The new phone features improved battery life and 
    an advanced camera system. Industry analysts predict strong sales 
    in the upcoming holiday season.
    """.strip()
    
    # 第一次生成（不帶 feedback）
    print("\n[Step 1] Generating fake news without feedback...")
    fake_v1 = generator(
        title=sample_title,
        content=sample_content
    )
    print(f"\nResult:\n{fake_v1}")
    
    # 第二次生成（帶 feedback）
    print("\n[Step 2] Regenerating with feedback...")
    fake_v2 = generator(
        title=sample_title,
        content=sample_content,
        feedback_prompt="Avoid 'revolutionary', 'game-changing', 'insider sources'. Use more subtle language."
    )
    print(f"\nResult with feedback:\n{fake_v2}")


if __name__ == "__main__":
    # 可以選擇執行 main() 或 demo_generator_function()
    main()
    # demo_generator_function()
