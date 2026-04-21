import requests
import cv2
import numpy as np
import os
import sys

# 通信チャネルを汚さないための設定
class Logger(object):
    def write(self, message):
        sys.stderr.write(message)
    def flush(self):
        sys.stderr.flush()
sys.stdout = Logger()

STASH_PORT = os.environ.get("STASH_PORT", "9999")
STASH_URL = f"http://127.0.0.1:{STASH_PORT}/graphql"
TARGET_TAG = "Mosaic"

def stash_query(query, variables=None):
    headers = {"Content-Type": "application/json"}
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
    try:
        # 12万件のデータを受け取るためタイムアウトを長めに設定
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=120)
        return res.json().get('data')
    except Exception as e:
        sys.stderr.write(f"Query Failed: {e}\n")
        return None

def is_mosaic(path):
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        img = cv2.resize(img, (500, 500))
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=10)
        if lines is None: return False
        grid_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 2 or abs(line[0][1] - line[0][3]) < 2)
        return grid_lines > 50
    except:
        return False

def main():
    sys.stderr.write("--- Starting Mosaic Detector (Final Edition) ---\n")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None) if data else None
    if not tag_id:
        sys.stderr.write(f"Error: タグ '{TARGET_TAG}' を作成してください。\n")
        return

    # 2. 12万件の画像パスを一括取得（paths { screenshot } を使用）
    sys.stderr.write("Fetching 120k image paths...\n")
    query = '{ allImages { id paths { screenshot } } }'
    i_data = stash_query(query)
    
    if not i_data:
        sys.stderr.write("Failed to fetch images. Data size might be too large.\n")
        return

    images = i_data.get('allImages', [])
    total = len(images)
    sys.stderr.write(f"Scanning {total} images. This may take a while...\n")

    # 3. 解析ループ
    detected_count = 0
    for count, img in enumerate(images, 1):
        img_id = img['id']
        # paths オブジェクトから実際のパスを取り出す
        path = img.get('paths', {}).get('screenshot')

        if is_mosaic(path):
            detected_count += 1
            sys.stderr.write(f"[{count}/{total}] Detected: {os.path.basename(path)}\n")
            
            # タグ更新
            u_query = 'mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }'
            stash_query(u_query, {"id": img_id, "tags": [tag_id]})
        
        # 1000枚ごとにログで生存確認
        if count % 1000 == 0:
            sys.stderr.write(f"Progress: {count}/{total} images processed...\n")

    sys.stderr.write(f"--- Task Completed! Found: {detected_count} ---\n")

if __name__ == "__main__":
    main()
