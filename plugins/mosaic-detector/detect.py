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
    if API_KEY: headers["ApiKey"] = API_KEY
    try:
        res = requests.post(STASH_URL, json={'query': query}, headers=headers)
        if res.status_code == 400:
            print(f"GraphQL 400 Error: {res.text}")
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
    print("--- Starting Mosaic Detector (Paginated Mode) ---")
    
    # 1. タグのIDを取得
    data = stash_query('{ allTags { id name } }')
    if not data: return
    tags = data.get('allTags', [])
    tag_id = next((t['id'] for t in tags if t['name'] == TARGET_TAG), None)

    if not tag_id:
        print(f"Error: タグ '{TARGET_TAG}' を作成してください。")
        return

    # 2. 画像リストを分割して取得・解析
    page = 1
    per_page = 1000  # 1000件ずつ取得
    detected_count = 0
    total_processed = 0

    while True:
        # 分割取得用のクエリ（findImages を使用）
        query = f'''
        {{
          findImages(image_filter: {{}}, filter: {{ page: {page}, per_page: {per_page} }}) {{
            count
            images {{
              id
              path
            }}
          }}
        }}
        '''
        print(f"Fetching page {page}...")
        i_data = stash_query(query)
        if not i_data: break

        find_data = i_data.get('findImages', {})
        images = find_data.get('images', [])
        total_count = find_data.get('count', 0)

        if not images:
            break

        for img in images:
            total_processed += 1
            if is_mosaic(img['path']):
                detected_count += 1
                print(f"[{total_processed}/{total_count}] Detected: {os.path.basename(img['path'])}")
                
                # タグ付与
                u_query = 'mutation { imageUpdate(input: { id: "' + img['id'] + '", tag_ids: ["' + tag_id + '"] }) { id } }'
                stash_query(u_query)

        if total_processed >= total_count:
            break
        page += 1

    print(f"--- Finished! Processed: {total_processed}, Detected: {detected_count} ---")

if __name__ == "__main__":
    main()
