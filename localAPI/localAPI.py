from flask import Flask, request, jsonify
from flask_cors import CORS
from search_api import extract_article_main
from chunking_api import chunk_text_main

app = Flask(__name__)
CORS(app)

@app.route("/extract", methods=["POST"])
def extract_route():
    try:
        data = request.get_json()
        urls = data.get("url", [])
        if isinstance(urls, str):
            urls = [urls]  # 包裝成 list
        return jsonify(extract_article_main(urls))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chunking", methods=["POST"])
def chunking_route():
    try:
        data = request.get_json()
        text = data.get("text", "")
        threshold = float(data.get("threshold", 0.55))
        return jsonify(chunk_text_main(text, threshold))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
