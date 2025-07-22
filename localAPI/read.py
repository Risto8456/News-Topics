# read.py
# -*- coding: utf-8 -*-
import requests
from readability import Document
from bs4 import BeautifulSoup
import re

# 定義 Article 資料結構
class Article:
    def __init__(self, url, title="", content=""):
        self.url = url
        self.title = title
        self.content = content

# 過濾出包含中文的段落
def filter_chinese_paragraphs(text):
    lines = [line for line in text.split('\n') if re.search(r'[\u4e00-\u9fff]', line)]
    return '\n'.join(lines)

# 嘗試不同解碼方式處理 HTML 字串
def robust_decode(response):
    try:
        return response.content.decode('utf-8')
    except Exception:
        pass
    try:
        return response.content.decode('big5', errors='ignore')
    except Exception:
        pass
    try:
        enc = response.apparent_encoding or 'utf-8'
        return response.content.decode(enc, errors='ignore')
    except Exception:
        pass
    raise ValueError("decode error: 無法正確解碼此網頁")

# 核心函式：輸入 URL，回傳 Article 物件（含 title 與 content）
def get_article(url):
    if url.lower().endswith('.pdf'):
        raise ValueError("PDF URLs not supported")

    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=30)
    res.raise_for_status()
    html_string = robust_decode(res)

    doc = Document(html_string)
    main_article_html = doc.summary()
    title = doc.title()
    soup_main = BeautifulSoup(main_article_html, 'html.parser')
    main_article_text = soup_main.get_text(separator='\n').strip()
    filtered_text = filter_chinese_paragraphs(main_article_text)

    if filtered_text:
        lines = [line for line in filtered_text.split('\n') if line.strip()]
        excerpt = "\n".join(lines[:10])
        return Article(url=url, title=title, content=excerpt)
    else:
        return Article(url=url, title=title, content="")
