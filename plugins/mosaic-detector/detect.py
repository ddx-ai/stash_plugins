import requests
import cv2
import numpy as np
import os
import sys

# 通信チャネル保護（標準出力をエラー出力へ）
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
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=60)
        if res.status_code != 200:
            return None
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
    sys.stderr.write("--- Starting Paginated Image Scan (120k+) ---\n")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None) if data else None
    if not tag_id:
        sys.stderr.write(f"Error: タグ '{TARGET_TAG}' を作成してください。\n")
        return

    # 2. ページネーションで500件ずつ取得
    page = 1
    per_page = 500 
    total_processed = 0
    detected_count = 0

    while True:
        # findImages クエリを使用。filter引数の型をシンプルに指定。
        query = '''
        query FindImages($f: FindFilterType) {
          findImages(filter: $f) {
            count
            images {
              id
              paths { screenshot }
            }
          }
        }
        '''
        variables = {"f": {"page": page, "per_page": per_page}}
        
        i_data = stash_query(query, variables)
        if not i_data:
            sys.stderr.write(f"Page {page} でエラーが発生しました。終了します。\n")
            break

        res = i_data.get('findImages', {})
        images = res.get('images', [])
        total_count = res.get('count', 0)

        if not images:
            break

        for img in images:
            total_processed += 1
            path = img.get('paths', {}).get('screenshot')
            
            if is_mosaic(path):
                detected_count += 1
                sys.stderr.write(f"[{total_processed}/{total_count}] Detected: {os.path.basename(path)}\n")
                
                # タグ付与
                u_query = 'mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }'
                stash_query(u_query, {"id": img['id'], "tags": [tag_id]})

        sys.stderr.write(f"Progress: {total_processed}/{total_count} images...\n")
        
        if total_processed >= total_count:
            break
        page += 1

    sys.stderr.write(f"--- All Done! Found: {detected_count} ---\n")

if __name__ == "__main__":
    main()
