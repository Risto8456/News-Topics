# searchapi_TLE.py
def extract_article_main(urls):
    from read import get_article
    results = []

    for url in urls:
        try:
            article = get_article(url)
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
