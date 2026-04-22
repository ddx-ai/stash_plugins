import requests
import cv2
import numpy as np
import os
import sys
import json

STASH_PORT = os.environ.get("STASH_PORT", "9999")
STASH_URL = f"http://127.0.0.1:{STASH_PORT}/graphql"
TAG_MOSAIC = "Mosaic"
TAG_NO_MOSAIC = "NoMosaic"

class Logger(object):
    def write(self, message):
        sys.stderr.write(message)
    def flush(self):
        sys.stderr.flush()
sys.stdout = Logger()

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

def get_config():
    """StashのSettingsからReCheckModeを取得"""
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            plugins = data.get('server_config', {}).get('plugins', {})
            config = plugins.get('Mosaic Detector', {})
            # ここでデフォルト値を指定（辞書のgetメソッドの第2引数）
            return config.get('ReCheckMode', False)
    except:
        pass
    return False

def is_mosaic(path):
    """
    高精度版：解像度が低い、または目が細かいモザイクを検出
    """
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path)
        if img is None: return False
        
        # 1. 小さなモザイクを潰さないよう、高解像度で再描画
        img_res = cv2.resize(img, (1024, 1024))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        # 2. 適応型二値化でタイルの輪郭を強調 (サムネイル対策)
        # 影や明るさに左右されず、エッジを抽出
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. モルフォロジー演算で細かいノイズを除去しつつタイルを繋ぐ
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # 4. 輪郭抽出
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mosaic_candidate_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 240px相当だと、タイルの1粒はかなり小さい(5-500px程度)
            if 5 < area < 800: 
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                # 四角形に近いかどうかを判定 (近似の精度を上げる)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                # 四角形（タイル）特有の形状をカウント
                if 4 <= len(approx) <= 8:
                    mosaic_candidate_count += 1
                    
        # 判定しきい値
        # 240pxで判別しにくい細かいものはタイルの数自体が多くなる傾向にあります
        # ここでは50個以上の「タイルの粒」が見つかればモザイクと判定します
        return mosaic_candidate_count > 50

    except Exception as e:
        sys.stderr.write(f"Analyze Error: {e}\n")
        return False

def main():
    re_check_mode = get_config()
    sys.stderr.write(f"--- Mode: {'RE-CHECK' if re_check_mode else 'NEW ONLY'} ---\n")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    m_id, n_id = tags_map.get(TAG_MOSAIC), tags_map.get(TAG_NO_MOSAIC)

    if not m_id or not n_id:
        sys.stderr.write("Error: Mosaic/NoMosaic tags missing.\n")
        return

    # 2. 【APIエラー回避策】まず全画像のIDとタグ情報を一括取得
    # 複雑なフィルターをAPIに投げず、Python側で仕分けます
    sys.stderr.write("Fetching image list from Stash...\n")
    res = stash_query('{ allImages { id tags { id } } }')
    all_images = res.get('allImages', []) if res else []
    
    images = []
    if re_check_mode:
        images = all_images
    else:
        # MosaicもNoMosaicも持っていない画像だけを抽出（Python側で判定）
        for img in all_images:
            existing_tag_ids = [t['id'] for t in img.get('tags', [])]
            if m_id not in existing_tag_ids and n_id not in existing_tag_ids:
                images.append(img)

    total = len(images)
    sys.stderr.write(f"Target: {total} images (Filtered from {len(all_images)} total).\n")

    # 3. ループ処理
    for count, item in enumerate(images, 1):
        img_id = item['id']
        # ファイルパス取得
        detail = stash_query('query G($id: ID){ findImage(id: $id){ files { path } tags { id } } }', {"id": img_id})
        if not detail or not detail['findImage']: continue
        
        img_data = detail['findImage']
        path = img_data['files'][0]['path']
        current_tag_ids = [t['id'] for t in img_data.get('tags', [])]

        is_m = is_mosaic(path)
        new_tag_id = m_id if is_m else n_id
        
        clean_tag_ids = [tid for tid in current_tag_ids if tid not in [m_id, n_id]]
        updated_tag_ids = clean_tag_ids + [new_tag_id]

        if set(current_tag_ids) != set(updated_tag_ids):
            stash_query('mutation U($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }', 
                        {"id": img_id, "tags": updated_tag_ids})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} done.\n")

    sys.stderr.write("--- Scan Finished --- \n")

if __name__ == "__main__":
    main()
