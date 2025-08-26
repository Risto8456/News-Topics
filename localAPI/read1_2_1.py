# read.py
# -*- coding: utf-8 -*-
# 需求套件：
# pip install requests readability-lxml beautifulsoup4 sentence-transformers

'''
修改 read1_2.py：強制走本地路徑 + 啟動就「暖機驗證」
'''
import re
import requests
import numpy as np
from readability import Document
from bs4 import BeautifulSoup

# 語意比對用（延遲載入，避免每次匯入就初始化模型）
import torch
import os
from sentence_transformers import SentenceTransformer, util

MODEL_DIR = os.getenv("EMBED_MODEL_PATH", r"C:\Users\User\OneDrive\桌面\專題\hf_models\bge-base-zh-v1.5")
_MODEL = None

def _ensure_weights_loaded(model):
    # 檢查是否有 meta tensors（空殼）
    if any(getattr(p, "is_meta", False) or (getattr(p, "device", None) and p.device.type == "meta")
           for p in model.parameters()):
        raise RuntimeError("模型權重未正確載入（偵測到 meta tensors）。")

def _warmup(model):
    # 啟動時先跑一次最小 encode，確定真的能用
    _ = model.encode(["ping"], convert_to_tensor=True)

def _get_model():
    global _MODEL
    if _MODEL is None:
        # 只用「本地目錄」；不要用 repo 名稱（會走線上）
        _MODEL = SentenceTransformer(MODEL_DIR, device="cpu")
        _ensure_weights_loaded(_MODEL)
        _warmup(_MODEL)
    return _MODEL


# ---- Article ----
class Article:
    def __init__(self, url, title="", content=""):
        self.url = url
        self.title = title
        self.content = content
    def __str__(self): # 測試輸出用
        return f"Article:\n  url: {self.url}\n  title: {self.title}\n  content: {self.content}"

# ---- 多步驟解碼 ----
def decode_response_text(response):
    try:
        return response.content.decode('utf-8', errors='ignore')
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

# ---- 只保留中文 ＋ 挑選高相似段落 ----
def filter_chinese_paragraphs(text, ref_paragraph=None):
    """
    text            : 文章全文（純文字）
    ref_paragraph   : 「原文段落」做語意比對，回傳較相近的段落
    """
    # 1) 只留中文（含標點）、數字、英文字母、常見符號
    #    這裡保留常用標點，避免句子被黏在一起
    keep = re.findall(r'[A-Za-z0-9\u4e00-\u9fff，。！？、：《》〈〉（）()\[\]【】：「」；—\-–—…‧·～~％%．。,:;!?\'" ]', text)
    filtered = ''.join(keep)

    # 若沒有提供 ref_paragraph 或內容為空，維持舊行為：直接回傳「只含中文」的結果
    if not ref_paragraph or not filtered.strip():
        return filtered

    # 2) 單純用句子分塊
    sentences = re.split(r'(?<=[。！？])\s*', filtered.strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return ""

    # 3) 與「原文段落」做餘弦相似度
    model = _get_model()
    ref_vec = model.encode(ref_paragraph, convert_to_tensor=True)         # (d,)
    sent_vecs = model.encode(sentences, convert_to_tensor=True)           # (N, d)
    sims = util.cos_sim(sent_vecs, ref_vec).squeeze(-1)                   # (N,)

    # debug : 列出每個句子的餘弦相似度
    """
    print("=== filter_chinese_paragraphs: sentence & sim ===")
    for s, sc in zip(sentences, sims.tolist()):
        print(f"{sc:.4f}\t{s}")
    """
    
    # 挑選相似度較高的句子
    picked_text = _pick_sentences_with_mmr(
        sentences, sent_vecs, sims,
        k_max=2,         # 想更短就降到 4~5
        percentile=0.90, # 想更嚴格就調到 0.90
        lambda_div=0.7,  # 0.6~0.8 常見
        max_chars=500    # 視需求調整
    )
    return picked_text

# 挑選，做控量＋去重
def _pick_sentences_with_mmr(sentences, sent_vecs, sims,
                             k_max=6, percentile=0.85,  # 控量
                             lambda_div=0.7,            # 多樣性權重 (0~1)
                             max_chars=1500):           # 字數上限
    # 1) 動態門檻（例如取相似度分佈的 85 分位）
    s = sims.detach().cpu().numpy().astype(float)
    thr = np.quantile(s, percentile)
    cand = np.where(s >= thr)[0]
    if cand.size == 0:
        cand = np.array([int(np.argmax(s))])

    # 2) 句間相似度矩陣（用於 MMR 去重）
    ss = util.cos_sim(sent_vecs, sent_vecs).cpu().numpy().astype(float)

    # 3) MMR Top-K（先選最相似，再兼顧多樣性）
    sel = [int(cand[np.argmax(s[cand])])]
    while len(sel) < min(k_max, len(cand)):
        rest = [i for i in cand if i not in sel]
        if not rest:
            break
        # MMR: λ·sim(query, i) − (1−λ)·max_j sim(i, j), j∈sel
        mmr_scores = lambda_div * s[rest] - (1 - lambda_div) * np.max(ss[rest][:, sel], axis=1)
        sel.append(rest[int(np.argmax(mmr_scores))])

    sel = sorted(sel)

    # 4) 合併相鄰句子，提升可讀性
    merged = []
    buf = [sel[0]]
    for i in sel[1:]:
        if i == buf[-1] + 1:
            buf.append(i)
        else:
            merged.append(''.join(sentences[j] for j in buf))
            buf = [i]
    merged.append(''.join(sentences[j] for j in buf))

    # 5) 字數上限裁切
    out, acc = [], 0
    for seg in merged:
        if acc + len(seg) > max_chars:
            break
        out.append(seg)
        acc += len(seg) + 1

    return "\n".join(out)

# ---- 主要函式：輸入 URL → 回傳 Article ----
def get_article(url, ref_paragraph=None):
    if url.lower().endswith('.pdf'):
        raise ValueError("PDF URLs not supported")

    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=30)
    res.raise_for_status()
    html_string = decode_response_text(res)

    # Readability 擷取主文 HTML 與標題
    doc = Document(html_string)
    main_article_html = doc.summary()
    title = doc.title()

    # 轉成純文字
    soup_main = BeautifulSoup(main_article_html, 'html.parser')
    main_article_text = soup_main.get_text(separator='\n').strip()
    
    # 新增：超過 40000 字就直接返回
    if len(main_article_text) > 40000:
        raise ValueError("There is too much content on the web page to judge")
    
    # 這裡導入 ref_paragraph；若沒帶，行為與舊版相同（只保留中文）
    filtered_text = filter_chinese_paragraphs(
        main_article_text,
        ref_paragraph=ref_paragraph
    )

    if filtered_text:
        lines = [line for line in filtered_text.split('\n') if line.strip()]
        excerpt = "\n".join(lines)
        return Article(url=url, title=title, content=excerpt)
    else:
        return Article(url=url, title=title, content="")
    
if __name__ == "__main__":
    # 範例
    url = "https://erc.com.tw/%E4%BA%BA%E8%B3%87%E6%96%B0%E8%81%9E"
    ref_paragraph = "有義烏商家表示，關稅下降的消息一出，「朋友圈都炸掉了，都在說要開幹了」、「接下來15天肯定是要大幹的」，要抓緊工廠端生產、出貨，也要加大接單的投入，以留出足夠的反應時間。"
    result = get_article(url, ref_paragraph)
    print(result)