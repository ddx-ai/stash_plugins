import requests
import cv2
import numpy as np
import os
import sys

# --- 設定 ---
# ブラウザでStashを開いている時のURLに合わせてください
STASH_URL = "http://127.0.0.1:9999/graphql" 
API_KEY = ""  # 設定している場合のみ入力
TARGET_TAG = "Mosaic" 
THRESHOLD = 50  # 直線検出のしきい値。誤検知が多いなら上げる、漏れが多いなら下げる

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
    """画像全体に格子状のパターンがあるか判定"""
    if not os.path.exists(path):
        return False
    
    # 画像をグレースケールで読み込み
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    
    # 解析速度向上のためリサイズ
    img = cv2.resize(img, (500, 500))
    
    # エッジ検出
    edges = cv2.Canny(img, 50, 150)
    
    # ハフ変換で直線を検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
    
    if lines is None:
        return False
    
    # 垂直・水平に近い線だけをカウント
    grid_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 2 or abs(y1 - y2) < 2:
            grid_lines += 1
            
    return grid_lines > THRESHOLD

def main():
    print(f"--- Mosaic Detection Task Started ---")

    # 1. タグの準備
    tag_data = stash_query('{ allTags { id name } }')
    if tag_data is None:
        print("Error: Stashへの接続に失敗しました。URLまたはAPIキーを確認してください。")
        return

    tag_id = next((t['id'] for t in tag_data.get('allTags', []) if t['name'] == TARGET_TAG), None)

    if not tag_id:
        print(f"Tag '{TARGET_TAG}' が見つかりません。新規作成します。")
        create_res = stash_query('''
            mutation($name: String!) {
                tagCreate(input: { name: $name }) { id }
            }
        ''', {"name": TARGET_TAG})
        if create_res:
            tag_id = create_res['tagCreate']['id']
            print(f"Created Tag ID: {tag_id}")
        else:
            print("Error: タグの作成に失敗しました。")
            return

   # --- 修正後の main 関数内 ---

    # 2. 画像一覧を取得
    print("画像リストを取得中...")
    # 'tags { id }' を一旦外し、最小構成でリクエストして 400 エラーを回避します
    image_data = stash_query('{ allImages { id path } }')
    
    if not image_data:
        print("画像データが取得できませんでした。")
        return

    images = image_data.get('allImages', [])
    total = len(images)
    print(f"Total images to scan: {total}")

    # 3. 解析とタグ付け
    count = 0
    detected_count = 0

    for img in images:
        count += 1
        img_id = img['id']
        path = img['path']
        
        # 解析実行
        if is_mosaic(path):
            detected_count += 1
            print(f"[{count}/{total}] Detected Mosaic: {os.path.basename(path)}")
            
            # タグを付与（imageUpdateをシンプルに実行）
            # ids をリスト形式で渡すのが Stash API のルールです
            stash_query('''
                mutation($img_id: ID!, $tag_ids: [ID!]) {
                    imageUpdate(input: { id: $img_id, tag_ids: $tag_ids }) { id }
                }
            ''', {"img_id": img_id, "tag_ids": [tag_id]})
        
        if count % 100 == 0:
            print(f"Progress: {count}/{total} images processed...")

    print(f"--- Finished ---")
    print(f"Scanned: {total}, Detected: {detected_count}")

if __name__ == "__main__":
    main()
