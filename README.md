## Workflows n8n 工作流
+ 全部下載
+ 執行 main 就好，其它是子工作流
+ n8n 須設定讀取/寫入磁碟功能
+ "Get 3 Divide Keywords" 最多需要 3 個 OpenAI Key
+ 有一些 HTTP 我是用 Gemini 一些是 OpenAI

## Local API 本地 API
+ 只要執行 localAPI.py 就行
+ n8n HTTP Request 節點使用時
  1. 將文章依語意分段：http://192.168.100.2:5000/chunking
  2. 網頁爬蟲抓取文字：http://192.168.100.2:5000/extract
## 餘弦相似度計算 `bge-base-zh-v1.5` model
+ 執行以下 python 程式碼下載
```py=
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-base-zh-v1.5",
                  local_dir="/models/bge-base-zh-v1.5",
                  local_dir_use_symlinks=False)
```
