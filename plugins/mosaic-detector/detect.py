import requests
import cv2
import numpy as np
import os

# --- 設定 ---
STASH_URL = "http://127.0.0.1:9999/graphql" 
API_KEY = "" 
TARGET_TAG = "Mosaic" 
THRESHOLD = 50 

def stash_query(query, variables=None):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["ApiKey"] = API_KEY
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
    try:
        res = requests.post(STASH_URL, json=payload, headers=headers)
        if res.status_code != 200:
            return None
        return res.json().get('data')
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def is_mosaic(path):
    """OpenCVによる格子状モザイクの判定"""
    if not os.path.exists(path): return False
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    
    # 500x500にリサイズして解析速度を一定にする
    img = cv2.resize(img, (500, 500))
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
    
    if lines is None: return False
    
    grid_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 垂直・水平に近い直線をカウント
        if abs(x1 - x2) < 2 or abs(y1 - y2) < 2:
            grid_lines += 1
    return grid_lines > THRESHOLD

def main():
    print("--- Starting Mosaic Detector ---")
    
    # 1. タグのIDを取得（なければ作成）
    data = stash_query('{ allTags { id name } }')
    tags = data.get('allTags', []) if data else []
    tag_id = next((t['id'] for t in tags if t['name'] == TARGET_TAG), None)

    if not tag_id:
        print(f"Tag '{TARGET_TAG}' を作成中...")
        c_data = stash_query('mutation($n: String!) { tagCreate(input: { name: $n }) { id } }', {"n": TARGET_TAG})
        if c_data:
            tag_id = c_data['tagCreate']['id']
        else:
            print("タグの作成に失敗しました。")
            return

    # 2. 画像リストを取得（成功実績のあるクエリ）
    print("画像リストを取得中...")
    i_data = stash_query('{ allImages { id path } }')
    if not i_data:
        print("画像リストが空、または取得できませんでした。")
        return

    images = i_data.get('allImages', [])
    total = len(images)
    print(f"解析開始: {total} 枚の画像をチェックします。")

    detected_count = 0
    for count, img in enumerate(images, 1):
        img_id = img['id']
        path = img['path']

        # 解析実行
        if is_mosaic(path):
            detected_count += 1
            print(f"[{count}/{total}] Detected: {os.path.basename(path)}")
            
            # タグを付与
            stash_query('''
                mutation($img_id: ID!, $tag_ids: [ID!]) {
                    imageUpdate(input: { id: $img_id, tag_ids: $tag_ids }) { id }
                }
            ''', {"img_id": img_id, "tag_ids": [tag_id]})
        
        # 100枚ごとに進捗表示
        if count % 100 == 0:
            print(f"Progress: {count}/{total} processed...")

    print(f"--- Finished ---")
    print(f"Total: {total}, Mosaic Tags Applied: {detected_count}")

if __name__ == "__main__":
    main()
