import requests
import cv2
import numpy as np
import os
import sys

# 通信チャネル保護
class Logger(object):
    def write(self, message):
        sys.stderr.write(message)
    def flush(self):
        sys.stderr.flush()
sys.stdout = Logger()

STASH_PORT = os.environ.get("STASH_PORT", "9999")
STASH_URL = f"http://127.0.0.1:{STASH_PORT}/graphql"
TAG_MOSAIC = "Mosaic"
TAG_NO_MOSAIC = "NoMosaic"

def stash_query(query, variables=None):
    headers = {"Content-Type": "application/json"}
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
    try:
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=20)
        return res.json().get('data')
    except:
        return None

def is_mosaic(path):
    """
    タイル構造（モザイク）の密度を解析するロジック
    """
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path)
        if img is None: return False
        
        # 1. 解析用の前処理（少し大きめに戻してタイルを認識しやすくする）
        img_res = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        # 2. ソーベルフィルタで垂直・水平の「色の段差」を抽出
        # 直線ではなく「色の変わり目」を面で捉える
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 絶対値をとって合成（エッジ画像）
        mag = np.hypot(sobelx, sobely)
        mag = np.uint8(mag / mag.max() * 255)
        
        # 3. モルフォロジー演算（ここがミソ）
        # モザイクの「点」を繋いで、格子の塊（クラスター）にする
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(mag, kernel, iterations=1)
        
        # 4. 輪郭抽出と「正方形度」のチェック
        # モザイクのタイルは小さな四角形の集まりなので、その形状をカウントする
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mosaic_candidate_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 小さすぎるノイズと、大きすぎる背景（4コマの枠など）を除外
            if 50 < area < 2000:
                # 形状の複雑さ（円形度に近い指標）を確認
                # モザイクのタイルが密集した領域は、ゴツゴツした塊になる
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                
                # 四角形に近い、または多角形の塊であればカウント
                if 4 <= len(approx) <= 12:
                    mosaic_candidate_count += 1

        # 5. 判定（密集度）
        # 512x512の画像内に、モザイクと思われるタイルの塊が一定数以上あるか
        # 床や縦じまは「巨大な1つの輪郭」になるため、個数は増えません。
        # 4コマ漫画も輪郭数は少ないです。
        return mosaic_candidate_count > 40

    except Exception as e:
        return False

def main():
    sys.stderr.write("--- Starting High-Density Mosaic Scan (Final Tuning) ---\n")
    
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    mosaic_id = tags_map.get(TAG_MOSAIC)
    no_mosaic_id = tags_map.get(TAG_NO_MOSAIC)

    if not mosaic_id or not no_mosaic_id:
        sys.stderr.write("Error: Mosaic or NoMosaic tags not found.\n")
        return

    all_data = stash_query('{ allImages { id } }')
    if not all_data: return
    images = all_data.get('allImages', [])
    total = len(images)

    for count, item in enumerate(images, 1):
        img_id = item['id']
        query = 'query GetImage($id: ID){ findImage(id: $id){ files { path } tags { id } } }'
        detail = stash_query(query, {"id": img_id})
        if not detail or not detail.get('findImage'): continue
        
        img_data = detail['findImage']
        path = img_data['files'][0]['path']
        current_tag_ids = [t['id'] for t in img_data.get('tags', [])]

        is_m = is_mosaic(path)
        new_tag_id = mosaic_id if is_m else no_mosaic_id
        
        updated_tag_ids = [tid for tid in current_tag_ids if tid not in [mosaic_id, no_mosaic_id]]
        updated_tag_ids.append(new_tag_id)

        if set(current_tag_ids) != set(updated_tag_ids):
            u_query = 'mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }'
            stash_query(u_query, {"id": img_id, "tags": updated_tag_ids})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} (Currently tagging {'Mosaic' if is_m else 'NoMosaic'})\n")

    sys.stderr.write("--- Scan Completed ---\n")

if __name__ == "__main__":
    main()
