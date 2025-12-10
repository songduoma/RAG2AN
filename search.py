import os
import requests

def search_google_custom_api(q, num_results=10):
    """
    使用 Google Custom Search API 進行搜尋
    需要設置環境變數: GOOGLE_CSE_API_KEY 和 GOOGLE_CSE_CX
    """
    api_key = os.environ.get("GOOGLE_CSE_API_KEY", "AIzaSyBQqWUfJ8Ql_3rIIjRrL6cMZN9Y0r9GJ20")
    cx = os.environ.get("GOOGLE_CSE_CX", "00f433b36691d4048")
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": q,
        "num": min(num_results, 10)  # Google Custom Search API 最多返回 10 個結果
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        print(f"[Google Search] Error: {e}")
        return {}

def search_serpapi(q):
    try:
        from serpapi import GoogleSearch  # lazy import to avoid hard dependency when google RAG is unused
    except Exception as exc:
        raise ImportError(
            "serpapi is required for Google RAG. Install `google-search-results` or set RAG_SOURCE to wiki/none."
        ) from exc

    params = {
        "api_key": os.environ["SERPAPI_API_KEY"],
        "engine": "google",
        "q": q,
        "location": "Austin, Texas, United States",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "num": "30"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def concat_snippets(organic_results):
    # filter out results without a snippet
    organic_results = [result for result in organic_results if 'snippet' in result]
    # filter out results from NBC News
    organic_results = [result for result in organic_results if 'NBC' not in result['source'] and 'NBC' not in result['title']]
    # filter out links containing 'fact'
    organic_results = [result for result in organic_results if 'fact' not in result['link']]
    organic_results = organic_results[:5]
    return '\n'.join([f'Title: {result["title"]}\nSource: {result["source"]}, {result["date"] if "date" in result else ""}\nContent: {result["snippet"]}' for result in organic_results])

def get_google_ctx(q):
    search_results = search_serpapi(q)
    if 'organic_results' in search_results:
        return concat_snippets(search_results['organic_results'])
    else:
        return ""

def get_google_custom_ctx(q, num_results=5):
    """
    使用 Google Custom Search API 獲取上下文
    返回格式化的搜尋結果字串
    """
    search_results = search_google_custom_api(q, num_results=num_results)
    if 'items' not in search_results or not search_results['items']:
        return ""
    
    context_lines = []
    for item in search_results['items'][:num_results]:
        title = item.get('title', '')
        snippet = item.get('snippet', '')
        link = item.get('link', '')
        
        # 格式化為與 Wikipedia 類似的格式
        line = f"- Title: {title}\n  Snippet: {snippet}"
        if link:
            line += f"\n  Link: {link}"
        context_lines.append(line)
    
    return "\n".join(context_lines)
