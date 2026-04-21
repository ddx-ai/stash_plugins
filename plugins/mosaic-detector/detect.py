import requests
import cv2
import numpy as np
import os
import sys

# 通信チャネル保護
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
        # IDリスト取得は時間がかかるためタイムアウトを長く
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=120)
        return res.json().get('data')
    except:
        return None

def is_mosaic(path):
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        img = cv2.resize(img, (500, 500))
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 80, 10)
        if lines is None: return False
        grid_lines = sum(1 for line in lines if abs(line[0][0]-line[0][2]) < 2 or abs(line[0][1]-line[0][3]) < 2)
        return grid_lines > 50
    except:
        return False

def main():
    sys.stderr.write("--- Starting Robust Scan (120k Images) ---\n")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None) if data else None
    if not tag_id:
        sys.stderr.write(f"Error: タグ '{TARGET_TAG}' を作成してください。\n")
        return

    # 2. 【実績あり】全IDだけを一括取得
    sys.stderr.write("Step 1: Fetching all image IDs (this will work)...\n")
    all_data = stash_query('{ allImages { id } }')
    if not all_data:
        sys.stderr.write("Failed to fetch ID list.\n")
        return

    images = all_data.get('allImages', [])
    total = len(images)
    sys.stderr.write(f"Step 2: Total {total} IDs found. Starting individual path scan...\n")

    # 3. 1件ずつパスを取得して解析（400エラーを物理的に回避）
    detected_count = 0
    for count, item in enumerate(images, 1):
        img_id = item['id']
        
        # 1枚ずつの取得なら Stash も絶対に拒否しません
        detail_query = 'query($id: ID!) { findImage(id: $id) { paths { screenshot } } }'
        detail_data = stash_query(detail_query, {"id": img_id})
        
        if not detail_data or not detail_data.get('findImage'):
            continue
            
        path = detail_data['findImage'].get('paths', {}).get('screenshot')

        if is_mosaic(path):
            detected_count += 1
            sys.stderr.write(f"[{count}/{total}] Detected: {os.path.basename(path)}\n")
            
            # タグ更新
            u_query = 'mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }'
            stash_query(u_query, {"id": img_id, "tags": [tag_id]})

        # 100枚ごとに生存報告
        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} images checked...\n")

    sys.stderr.write(f"--- Task Completed! Found: {detected_count} ---\n")

if __name__ == "__main__":
    main()
