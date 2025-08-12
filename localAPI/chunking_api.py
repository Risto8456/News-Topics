# chunking_api.py 根據語意將文章做分段
def chunk_text_main(text, threshold):
    from sentence_transformers import SentenceTransformer, util
    import re
    import numpy as np

    model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

    sentences = re.split(r'(?<=[。！？])\s*', text.strip())
    sentences = [s for s in sentences if s]

    # 語意合併相似句子成段落
    def semantic_chunking(sentences, model, similarity_threshold=0.55):
        if not sentences:
            return []

        embeddings = model.encode(sentences, convert_to_tensor=True)
        chunks = []
        current_chunk = sentences[0]

        for i in range(len(sentences) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            cosine_similarity = util.cos_sim(emb1, emb2).item()

            if cosine_similarity > similarity_threshold:
                current_chunk += " " + sentences[i + 1]
            else:
                chunks.append(current_chunk)
                current_chunk = sentences[i + 1]
        chunks.append(current_chunk)
        return chunks
    
    # 若 chunk 僅為標點符號，則合併回上一段
    def merge_orphan_punctuation(chunks):
        final_chunks = []
        for chunk in chunks:
            has_content = re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', chunk)
            if not has_content and final_chunks:
                final_chunks[-1] = final_chunks[-1].rstrip() + chunk
            else:
                final_chunks.append(chunk)
        return final_chunks

    merged_chunks = semantic_chunking(sentences, model, threshold)
    final_chunks = merge_orphan_punctuation(merged_chunks)
    embeddings = model.encode(final_chunks, convert_to_tensor=True)
    
    return [{"id": idx + 1, "text": chunk} for idx, chunk in enumerate(final_chunks)]

if __name__ == '__main__':
    with open("chunking_semantic_test/news.txt", "r", encoding="utf-8") as f:
        print(chunk_text_main(f.read(), 0.55))