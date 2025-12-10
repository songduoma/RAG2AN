#!/usr/bin/env python3
"""
測試 Google Custom Search API 整合
驗證 search_wikipedia 函數現在使用 Google Search API
"""

import os
import sys

# 設置環境變數（如果還沒設置）
os.environ.setdefault("GOOGLE_CSE_API_KEY", "AIzaSyBQqWUfJ8Ql_3rIIjRrL6cMZN9Y0r9GJ20")
os.environ.setdefault("GOOGLE_CSE_CX", "00f433b36691d4048")

# 導入修改後的函數
from generator import search_wikipedia

def test_search():
    """測試搜尋功能"""
    print("=" * 80)
    print("測試 Google Custom Search API 整合")
    print("=" * 80)
    print()
    
    # 測試查詢
    test_queries = [
        "large language model adversarial attacks",
        "artificial intelligence news",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[測試 {i}] 查詢: {query}")
        print("-" * 80)
        
        try:
            result = search_wikipedia(query, num_results=3, verbose=True)
            
            if result:
                print("\n✓ 搜尋成功！")
                print("\n結果:")
                print(result)
            else:
                print("\n⚠ 未返回結果")
                
        except Exception as e:
            print(f"\n✗ 發生錯誤: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    test_search()
