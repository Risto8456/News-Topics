# read.py
# -*- coding: utf-8 -*-
# 需求套件：
# pip install requests readability-lxml beautifulsoup4 sentence-transformers

import re
import requests
from readability import Document
from bs4 import BeautifulSoup

# 語意比對用（延遲載入，避免每次匯入就初始化模型）
import torch
from sentence_transformers import SentenceTransformer, util

# 使用你現有的 chunking 函式（依你給的 chunking_api.py）
# 預期回傳格式： [{"id": 1, "text": "段落1"}, {"id": 2, "text": "段落2"}, ...]
from chunking_api import chunk_text_main

# ---- 懶載模型，確保與 chunking_api 一致（BAAI/bge-base-zh-v1.5）----
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-base-zh-v1.5")
    return _MODEL

# ---- Article 結構（維持舊介面）----
class Article:
    def __init__(self, url, title="", content=""):
        self.url = url
        self.title = title
        self.content = content

# ---- 多步驟解碼（維持原樣）----
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

# ---- 擴充後的「只保留中文＋語意挑選」----
def filter_chinese_paragraphs(text, ref_paragraph=None, merge_threshold=0.55, top_k=1):
    """
    text            : 文章全文（純文字）
    ref_paragraph   : 「原文段落」；若提供，會做語意比對，只回傳最相近的 chunk(s)
    merge_threshold : 傳給 chunking_api 的相鄰句合併門檻（與你現有程式一致）
    top_k           : 回傳與 ref_paragraph 最接近的前 k 個 chunk（預設 1）
    """

    # 2) 舊行為：先只保留含中文（CJK）的行
    lines = [line for line in text.split('\n') if re.search(r'[\u4e00-\u9fff]', line)]
    filtered = '\n'.join(lines)

    # 若沒有 ref_paragraph，就維持舊行為：直接回傳「只含中文」的結果
    if not ref_paragraph or not filtered.strip():
        return filtered

    # 3) 將「只含中文的文本」交給 chunking_api 做「語意分段」
    #    預期回傳格式： [{"id":..., "text":"..."}...]
    try:
        chunk_items = chunk_text_main(filtered, merge_threshold)
    except Exception as e:
        # 若 chunking 出錯，退回舊行為
        return filtered

    # 取出純文字段落
    chunks = []
    for it in chunk_items:
        if isinstance(it, dict) and "text" in it and it["text"]:
            chunks.append(it["text"])
        elif isinstance(it, str) and it.strip():
            chunks.append(it)

    if not chunks:
        return ""  # 沒有可用段落

    # 4) 與「原文段落」做餘弦相似度，挑出最接近的 chunk(s)
    model = _get_model()
    ref_vec = model.encode(ref_paragraph, convert_to_tensor=True)          # (d,)
    chunk_vecs = model.encode(chunks, convert_to_tensor=True)              # (N, d)
    sims = util.cos_sim(chunk_vecs, ref_vec).squeeze(-1)                   # (N,)

    # 取相似度最高的前 k 個索引
    k = max(1, int(top_k))
    top_idx = torch.topk(sims, k=min(k, sims.shape[0])).indices.tolist()

    # 回傳挑中的段落（以 \n\n 串接）
    picked = [chunks[i] for i in top_idx]
    return "\n\n".join(picked)

# ---- 主要函式：輸入 URL → 回傳 Article（維持原樣，僅加上 ref_paragraph 參數）----
def get_article(url, ref_paragraph=None, top_k=1, merge_threshold=0.55):
    if url.lower().endswith('.pdf'):
        raise ValueError("PDF URLs not supported")

    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=30)
    res.raise_for_status()
    html_string = robust_decode(res)

    # Readability 擷取主文 HTML 與標題
    doc = Document(html_string)
    main_article_html = doc.summary()
    title = doc.title()

    # 轉成純文字
    soup_main = BeautifulSoup(main_article_html, 'html.parser')
    main_article_text = soup_main.get_text(separator='\n').strip()

    # 這裡導入 ref_paragraph；若沒帶，行為與舊版相同（只保留中文）
    filtered_text = filter_chinese_paragraphs(
        main_article_text,
        ref_paragraph=ref_paragraph,
        merge_threshold=merge_threshold,
        top_k=top_k
    )

    if filtered_text:
        # 與你原本相同：最多取前 10 行
        lines = [line for line in filtered_text.split('\n') if line.strip()]
        excerpt = "\n".join(lines[:10])
        return Article(url=url, title=title, content=excerpt)
    else:
        return Article(url=url, title=title, content="")
