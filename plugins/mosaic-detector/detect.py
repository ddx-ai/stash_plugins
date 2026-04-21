import requests
import cv2
import numpy as np
import os
import sys

# 標準出力をエラー出力へリダイレクト
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
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=30)
        # 400エラーが出た場合、何が原因か詳細を表示
        if res.status_code == 400:
            sys.stderr.write(f"GraphQL Detail: {res.text}\n")
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
    sys.stderr.write("--- Starting Mosaic Detector (Dynamic Field) ---\n")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    if not data: return
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None)
    if not tag_id:
        sys.stderr.write(f"Error: タグ '{TARGET_TAG}' を作成してください。\n")
        return

    # 2. 【変更】最も確実な ID と Path だけを取得
    # 'path' がダメなら、Stashの公式ドキュメントで推奨される最小構成を試します
    print("Fetching image list...")
    i_data = stash_query('{ allImages { id path } }') # ← もしここで再度400なら、'path' を消してテスト
    
    if not i_data:
        # 万が一 'path' がダメな時のためのバックアップ
        sys.stderr.write("Retrying with fallback schema...\n")
        i_data = stash_query('{ allImages { id } }')
    
    if not i_data: return
    images = i_data.get('allImages', [])
    sys.stderr.write(f"Scanning {len(images)} images...\n")

    # 3. 解析
    for img in images:
        img_id = img['id']
        # 'path' が存在しない、あるいは None の場合の処理
        path = img.get('path')
        
        # ログを出しすぎると重くなるので、見つけた時だけ表示
        if is_mosaic(path):
            sys.stderr.write(f"Found Mosaic: {os.path.basename(path)}\n")
            stash_query('''
                mutation($id: ID!, $tags: [ID!]) {
                    imageUpdate(input: { id: $id, tag_ids: $tags }) { id }
                }
            ''', {"id": img_id, "tags": [tag_id]})

    sys.stderr.write("--- Task Completed ---\n")

if __name__ == "__main__":
    main()
