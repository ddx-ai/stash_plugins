import requests
import cv2
import numpy as np
import os

# --- 設定 ---
STASH_URL = "http://127.0.0.1:9999/graphql" 
API_KEY = "" 
TARGET_TAG = "Mosaic" 
THRESHOLD = 50 

def stash_query(query):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["ApiKey"] = API_KEY
    # 変数(variables)を使わず、queryのみをJSONで送る
    payload = {'query': query}
    try:
        res = requests.post(STASH_URL, json=payload, headers=headers)
        if res.status_code != 200:
            print(f"HTTP Error: {res.status_code} - {res.text}")
            return None
        return res.json().get('data')
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def is_mosaic(path):
    if not os.path.exists(path): return False
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    img = cv2.resize(img, (500, 500))
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
    if lines is None: return False
    grid_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 2 or abs(line[0][1] - line[0][3]) < 2)
    return grid_lines > THRESHOLD

def main():
    print("--- Starting Mosaic Detector ---")
    
    # 1. タグのIDを取得
    data = stash_query('{ allTags { id name } }')
    if not data: return
    
    tags = data.get('allTags', [])
    tag_id = next((t['id'] for t in tags if t['name'] == TARGET_TAG), None)

    # 2. タグがなければ作成（クエリ内に直接名前を埋め込む）
    if not tag_id:
        print(f"Creating Tag: {TARGET_TAG}")
        c_query = 'mutation { tagCreate(input: { name: "' + TARGET_TAG + '" }) { id } }'
        c_data = stash_query(c_query)
        if c_data:
            tag_id = c_data['tagCreate']['id']
            print(f"Created Tag ID: {tag_id}")
        else:
            print("Failed to create tag.")
            return

    # 3. 画像リストを取得
    print("Fetching images...")
    i_data = stash_query('{ allImages { id path } }')
    if not i_data: return

    images = i_data.get('allImages', [])
    total = len(images)
    print(f"Scanning {total} images...")

    detected_count = 0
    for count, img in enumerate(images, 1):
        if is_mosaic(img['path']):
            detected_count += 1
            print(f"[{count}/{total}] Detected: {os.path.basename(img['path'])}")
            
            # タグ付与（クエリ内に直接IDを埋め込む）
            u_query = 'mutation { imageUpdate(input: { id: "' + img['id'] + '", tag_ids: ["' + tag_id + '"] }) { id } }'
            stash_query(u_query)
        
        if count % 100 == 0:
            print(f"Progress: {count}/{total} processed...")

    print(f"--- Finished! Detected: {detected_count} ---")

if __name__ == "__main__":
    main()
