#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
將指定資料夾中的 XX_final.json 檔案（例如 1_final.json, 2_final.json ...）
依數字順序串聯成一個 final.json。
"""

import os
import re
import json
import glob

# ======= 你要修改的路徑 =======
INPUT_DIR = r".\result1"   # 放 XX_final.json 的資料夾
OUTPUT_FILE = r".\result1_analysis\final.json"  # 輸出檔案路徑
# =============================

def numeric_key(path: str) -> int:
    """從檔名抽出數字作排序 key"""
    name = os.path.basename(path)
    m = re.match(r"(\d+)_final\.json$", name)
    return int(m.group(1)) if m else 10**9

def main():
    pattern = os.path.join(INPUT_DIR, "*_final.json")
    files = glob.glob(pattern)
    files.sort(key=numeric_key)

    data = []
    errors = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            data.append(obj)
        except Exception as e:
            errors.append({"file": os.path.basename(fp), "error": str(e)})

    # 寫出 final.json
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] 合併完成：{len(files)} 檔 → {OUTPUT_FILE}")
    if errors:
        print(f"[WARN] 讀取失敗 {len(errors)} 筆：")
        for e in errors:
            print(" -", e)

if __name__ == "__main__":
    main()
