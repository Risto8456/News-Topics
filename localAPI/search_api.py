# search_api.py
def extract_article_main(urls, ref_paragraph=None):
    from read import get_article
    results = []

    for url in urls:
        try:
            article = get_article(url, ref_paragraph, 3, 0.55)
            if article.content:
                results.append({
                    "url": url,
                    "title": article.title,
                    "content": article.content
                })
            else:
                results.append({
                    "url": url,
                    "title": "錯誤",
                    "content": "",
                    "error": "無法提取到主文內容"
                })
        except Exception as e:
            results.append({
                "url": url,
                "title": "錯誤",
                "content": "",
                "error": str(e)
            })

    return results
