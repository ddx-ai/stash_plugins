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
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=20)
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
    sys.stderr.write("--- Starting Scraper-Based Scan (120k) ---\n")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None) if data else None
    if not tag_id:
        sys.stderr.write(f"Error: タグ '{TARGET_TAG}' を作成してください。\n")
        return

    # 2. 全ID取得
    all_data = stash_query('{ allImages { id } }')
    if not all_data: return
    images = all_data.get('allImages', [])
    total = len(images)
    sys.stderr.write(f"Total {total} images. Starting scan...\n")

    detected_count = 0
    for count, item in enumerate(images, 1):
        img_id = item['id']
        
        # 3. スクレーパーと同じ形式でクエリ（findImage -> files -> path）
        query = """
        query GetPath($id: ID){
          findImage(id: $id){
            files {
              path
            }
          }
        }
        """
        detail = stash_query(query, {"id": img_id})
        
        if not detail or not detail.get('findImage'):
            continue
            
        # filesリストから最初の要素のpathを取得
        files = detail['findImage'].get('files', [])
        if not files: continue
        path = files[0].get('path')

        # 4. 解析
        if is_mosaic(path):
            detected_count += 1
            sys.stderr.write(f"[{count}/{total}] Detected: {os.path.basename(path)}\n")
            
            # タグ付与
            u_query = 'mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }'
            stash_query(u_query, {"id": img_id, "tags": [tag_id]})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} checked...\n")

    sys.stderr.write(f"--- Finished! Found: {detected_count} ---\n")

if __name__ == "__main__":
    main()
