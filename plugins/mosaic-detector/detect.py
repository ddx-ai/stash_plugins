import requests
import cv2
import numpy as np
import os
import sys

# Windows環境の文字化けによる通信エラーを防ぐ
sys.stdout.reconfigure(encoding='utf-8')

STASH_URL = "http://127.0.0.1:9999/graphql" 
API_KEY = "" 
TARGET_TAG = "Mosaic" 

def stash_query(query, variables=None):
    headers = {"Content-Type": "application/json"}
    if API_KEY: headers["ApiKey"] = API_KEY
    payload = {'query': query}
    if variables: payload['variables'] = variables
    try:
        res = requests.post(STASH_URL, json=payload, headers=headers)
        return res.json().get('data')
    except:
        return None

def is_mosaic(path):
    try:
        if not os.path.exists(path): return False
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
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    if not data: return
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None)
    if not tag_id: return

    # 2. 画像取得（フィールドを1つに絞って400エラーを物理的に回避）
    i_data = stash_query('{ allImages { id path } }')
    if not i_data: return
    images = i_data.get('allImages', [])

    # 3. 解析とタグ付け
    for img in images:
        if is_mosaic(img['path']):
            # 成功時も print を使わずに通信だけ行う
            stash_query('mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }', 
                        {"id": img['id'], "tags": [tag_id]})

if __name__ == "__main__":
    main()
