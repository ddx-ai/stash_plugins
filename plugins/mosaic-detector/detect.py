import requests
import cv2
import numpy as np
import os
#import stash_utils
# --- 設定 ---
STASH_URL = "http://localhost:9999/graphql"
API_KEY = "YOUR_API_KEY"  # APIキーがない場合は空でOK
TARGET_TAG = "Mosaic"     # 付与したいタグ名
THRESHOLD = 50            # 直線検出のしきい値（枚数。画像に合わせて調整）

headers = {"ApiKey": API_KEY} if API_KEY else {}

import os
import requests

def stash_query(query, variables=None):
    # プラグイン実行時にStashが提供するポートとAPIキーを環境変数から取得
    # これが最も確実な方法です
    port = os.environ.get("STASH_PORT", "9999")
    api_key = os.environ.get("STASH_API_KEY")
    
    # 127.0.0.1 (IPv4) を明示
    url = f"http://127.0.0.1:{port}/graphql"
    
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["ApiKey"] = api_key

    payload = {'query': query}
    if variables:
        payload['variables'] = variables

    try:
        res = requests.post(url, json=payload, headers=headers)
        
        # もしJSONじゃないものが返ってきたら、その内容をログに出して停止させる
        if "application/json" not in res.headers.get("Content-Type", ""):
            print(f"CRITICAL ERROR: Stash returned non-JSON response. HTTP {res.status_code}")
            print(f"Response snippet: {res.text[:200]}")
            return None
            
        return res.json().get('data')
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def is_full_mosaic(path):
    """画像全体に格子状のパターンがあるか判定"""
    if not os.path.exists(path): return False
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    
    # 縮小して処理を高速化
    img = cv2.resize(img, (500, 500))
    # エッジ検出
    edges = cv2.Canny(img, 50, 150)
    # ハフ変換で直線を検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
    
    if lines is None: return False
    
    # 垂直・水平に近い線だけをカウント
    grid_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 2 or abs(y1 - y2) < 2:
            grid_lines += 1
            
    return grid_lines > THRESHOLD

# 1. MosaicタグのIDを取得
tag_data = stash_query('{ allTags { id name } }')
tag_id = next((t['id'] for t in tag_data['allTags'] if t['name'] == TARGET_TAG), None)

if not tag_id:
    print(f"Error: Tag '{TARGET_TAG}' not found in Stash.")
    exit()

# 2. 全画像を取得（フィルタリングが必要ならvariablesを追加）
image_data = stash_query('{ allImages { id path } }')

print(f"Checking {len(image_data['allImages'])} images...")

for img in image_data['allImages']:
    if is_full_mosaic(img['path']):
        print(f"Detected Mosaic: {img['path']}")
        # 3. タグを付与
        stash_query('''
            mutation($img_id: ID!, $tag_id: [ID!]) {
                imageUpdate(input: { id: $img_id, tag_ids: $tag_id }) { id }
            }
        ''', {"img_id": img['id'], "tag_id": [tag_id]})
