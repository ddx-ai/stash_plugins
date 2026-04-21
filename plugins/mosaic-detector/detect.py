import requests
import cv2
import numpy as np
import os

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
        # 400エラーが出た場合に中身を詳しく表示する
        if res.status_code == 400:
            print(f"GraphQL Syntax Error (400): {res.text}")
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
    print("--- Mosaic Detector Start ---")
    
    # 1. タグの存在確認（単純な全件取得）
    data = stash_query('{ allTags { id name } }')
    if not data: return
    
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None)

    # 2. タグがなければ作成（構文を極限までシンプルに）
    if not tag_id:
        print(f"Creating tag: {TARGET_TAG}")
        create_query = 'mutation { tagCreate(input: { name: "' + TARGET_TAG + '" }) { id } }'
        c_data = stash_query(create_query)
        if c_data:
            tag_id = c_data['tagCreate']['id']
            print(f"Tag Created ID: {tag_id}")
        else:
            return

    # 3. 画像取得（件数が多い場合を想定し、filterを使用）
    print("Fetching images...")
    # 一度にすべて取らず、まずはIDとパスだけを取得
    i_data = stash_query('{ allImages { id path } }')
    if not i_data: return

    images = i_data.get('allImages', [])
    print(f"Scanning {len(images)} images...")

    for img in images:
        if is_mosaic(img['path']):
            print(f"Mosaic found: {os.path.basename(img['path'])}")
            # 更新クエリを文字列結合で作成（変数の型エラーを回避）
            update_query = 'mutation { imageUpdate(input: { id: "' + img['id'] + '", tag_ids: ["' + tag_id + '"] }) { id } }'
            stash_query(update_query)

    print("--- Finished ---")

if __name__ == "__main__":
    main()
