import requests
import cv2
import numpy as np
import os
import sys

# 【重要】標準出力をダミーに回し、Stashの通信チャネルを汚さないようにする
# ログを出したいときは sys.stderr.write() を使います
class Logger(object):
    def write(self, message):
        sys.stderr.write(message)
    def flush(self):
        sys.stderr.flush()

sys.stdout = Logger()

# 環境変数からポートを自動取得（デフォルト9999）
STASH_PORT = os.environ.get("STASH_PORT", "9999")
STASH_URL = f"http://127.0.0.1:{STASH_PORT}/graphql"
TARGET_TAG = "Mosaic" 

def stash_query(query, variables=None):
    # APIキーがクリアされたので、ヘッダーも最小構成に
    headers = {"Content-Type": "application/json"}
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
        
    try:
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=30)
        if res.status_code != 200:
            sys.stderr.write(f"HTTP {res.status_code}: {res.text}\n")
            return None
        return res.json().get('data')
    except Exception as e:
        sys.stderr.write(f"Query Failed: {e}\n")
        return None

def is_mosaic(path):
    if not os.path.exists(path): return False
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    img = cv2.resize(img, (500, 500))
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=10)
    if lines is None: return False
    grid_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 2 or abs(line[0][1] - line[0][3]) < 2)
    return grid_lines > 50

def main():
    sys.stderr.write("--- Starting Mosaic Detector ---\n")
    
    # 1. タグIDの取得
    data = stash_query('{ allTags { id name } }')
    if not data: return
    
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None)
    
    if not tag_id:
        sys.stderr.write(f"Tag '{TARGET_TAG}' not found. Please create it manually.\n")
        return

    # 2. 画像リストの取得（1.9MBのデータもこれで通るはずです）
    i_data = stash_query('{ allImages { id path } }')
    if not i_data: return
    images = i_data.get('allImages', [])
    
    sys.stderr.write(f"Scanning {len(images)} images...\n")

    # 3. 解析とタグ付け
    for img in images:
        if is_mosaic(img['path']):
            sys.stderr.write(f"Found: {os.path.basename(img['path'])}\n")
            # タグ付与
            stash_query('''
                mutation($id: ID!, $tags: [ID!]) {
                    imageUpdate(input: { id: $id, tag_ids: $tags }) { id }
                }
            ''', {"id": img['id'], "tags": [tag_id]})

    sys.stderr.write("--- Task Completed ---\n")

if __name__ == "__main__":
    main()
