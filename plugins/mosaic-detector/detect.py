import requests
import cv2
import numpy as np
import os
import sys
import json

# --- 通信設定とタグ名 ---
STASH_PORT = os.environ.get("STASH_PORT", "9999")
STASH_URL = f"http://127.0.0.1:{STASH_PORT}/graphql"
TAG_MOSAIC = "Mosaic"
TAG_NO_MOSAIC = "NoMosaic"

# ログ出力用
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
    except Exception as e:
        sys.stderr.write(f"Query Error: {e}\n")
        return None

def get_config():
    """StashのSettings画面からの設定値を取得"""
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            # plugin名 'Mosaic Detector' の下の 'ReCheckMode' を参照
            return data.get('server_config', {}).get('plugins', {}).get('Mosaic Detector', {}).get('ReCheckMode', False)
    except:
        pass
    return False

def is_mosaic(path):
    """タイル密度解析によるモザイク判定ロジック"""
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path)
        if img is None: return False
        
        # 解析サイズを統一
        img_res = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        # エッジ(色の段差)抽出
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        mag = np.uint8(mag / (mag.max() if mag.max() > 0 else 1) * 255)
        
        # モルフォロジー（膨張）でタイルの粒を塊にする
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(mag, kernel, iterations=1)
        
        # 輪郭（塊）の数をカウント
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mosaic_candidate_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # モザイクタイルらしいサイズの塊を拾う
            if 40 < area < 3000:
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                # 四角に近い多角形の集合体か
                if 4 <= len(approx) <= 12:
                    mosaic_candidate_count += 1
                    
        # 判定しきい値 (35個以上の島があればモザイクとみなす)
        return mosaic_candidate_count > 35
    except Exception as e:
        return False

def main():
    # 1. 設定の読み込み
    re_check_mode = get_config()
    mode_str = "RE-CHECK ALL" if re_check_mode else "NEW IMAGES ONLY"
    sys.stderr.write(f"--- Starting Scan Mode: {mode_str} ---\n")
    
    # 2. タグID取得（なければ作成を促す）
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    
    m_id = tags_map.get(TAG_MOSAIC)
    n_id = tags_map.get(TAG_NO_MOSAIC)

    if not m_id or not n_id:
        sys.stderr.write(f"Error: '{TAG_MOSAIC}' または '{TAG_NO_MOSAIC}' タグが見つかりません。\n")
        return

    # 3. 対象画像の取得
    if re_check_mode:
        res = stash_query('{ allImages { id } }')
        images = res.get('allImages', []) if res else []
    else:
        q = 'query Unchecked($ids: [ID!]) { findImages(image_filter: { tags: { modifier: NOT_INCLUDES, value: $ids } }) { images { id } } }'
        res = stash_query(q, {"ids": [m_id, n_id]})
        images = res['findImages']['images'] if res else []

    total = len(images)
    sys.stderr.write(f"Targeting {total} images.\n")

    # 4. ループ処理
    for count, item in enumerate(images, 1):
        img_id = item['id']
        detail = stash_query('query Get($id: ID){ findImage(id: $id){ files { path } tags { id } } }', {"id": img_id})
        if not detail or not detail['findImage']: continue
        
        img_data = detail['findImage']
        path = img_data['files'][0]['path']
        current_tag_ids = [t['id'] for t in img_data.get('tags', [])]

        # 判定
        is_m = is_mosaic(path)
        new_tag_id = m_id if is_m else n_id
        
        # 既存の判定タグを除去して新しいのを追加（他のタグは維持）
        clean_tag_ids = [tid for tid in current_tag_ids if tid not in [m_id, n_id]]
        updated_tag_ids = clean_tag_ids + [new_tag_id]

        # 変更がある場合のみStashを更新
        if set(current_tag_ids) != set(updated_tag_ids):
            stash_query('mutation Update($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }', 
                        {"id": img_id, "tags": updated_tag_ids})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total}...\n")

    sys.stderr.write("--- Task Completed ---\n")

if __name__ == "__main__":
    main()
