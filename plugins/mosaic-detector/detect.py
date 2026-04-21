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
    
    # 以前の文字列結合をやめ、標準的な dict 構造に戻します
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
        
    try:
        res = requests.post(STASH_URL, json=payload, headers=headers)
        if res.status_code == 400:
            # 400エラーの場合、Stash側が「どこがダメか」をJSONで返しているのでそれを表示
            print(f"!!! GraphQL Error (400) !!!")
            print(f"Detail: {res.text}")
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
    print("--- Mosaic Detector Engine Start ---")
    
    # 1. タグのIDを取得
    data = stash_query('{ allTags { id name } }')
    if not data: return
    tags = data.get('allTags', [])
    tag_id = next((t['id'] for t in tags if t['name'] == TARGET_TAG), None)

    if not tag_id:
        print(f"Error: Stashで '{TARGET_TAG}' タグを手動作成してください。")
        return

    # 2. 分割取得 (公式ブラウザ版と同じスキーマを使用)
    page = 1
    per_page = 1000
    total_processed = 0

    while True:
        # findImages の引数を変数を介して渡す、最も標準的な書き方
        query = '''
        query FindImages($filter: FindFilterType) {
          findImages(filter: $filter) {
            count
            images {
              id
              path
            }
          }
        }
        '''
        variables = {
            "filter": {
                "page": page,
                "per_page": per_page
            }
        }
        
        i_data = stash_query(query, variables)
        if not i_data: break

        find_res = i_data.get('findImages', {})
        images = find_res.get('images', [])
        total_count = find_res.get('count', 0)

        if not images:
            break

        for img in images:
            total_processed += 1
            if is_mosaic(img['path']):
                print(f"Detected: {os.path.basename(img['path'])}")
                
                # タグ更新 (変数を使い、エスケープ漏れを防ぐ)
                u_query = '''
                mutation ImageUpdate($id: ID!, $tag_ids: [ID!]) {
                  imageUpdate(input: { id: $id, tag_ids: $tag_ids }) { id }
                }
                '''
                u_vars = {"id": img['id'], "tag_ids": [tag_id]}
                stash_query(u_query, u_vars)

        if total_processed >= total_count:
            break
        page += 1

    print(f"--- All Done: {total_processed} images checked ---")

if __name__ == "__main__":
    main()
