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
    グリッド解析による高精度モザイク判定
    普通の画像（高精細なテクスチャ）をスルーし、人工的なモザイクのみを狙い撃ちします
    """
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path)
        if img is None: return False
        
        # 1. 解析用サイズ
        img_res = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        # 2. エッジの方向性を解析（モザイクは水平・垂直のエッジが極端に多い）
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 3. 勾配の強さを計算
        mag = np.hypot(sobelx, sobely)
        mag = np.uint8(mag / (mag.max() if mag.max() > 0 else 1) * 255)
        
        # 二値化して塊を抽出
        _, thresh = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mosaic_cells = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 小さなタイルの粒（5〜400ピクセル程度）をターゲットにする
            if 10 < area < 400:
                # 形状の矩形度（どれだけ正方形・長方形に近いか）を確認
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                extent = float(area) / rect_area # 占有率
                aspect_ratio = float(w) / h # アスペクト比
                
                # 「四角に近い」かつ「極端に細長くない」ものを抽出
                if extent > 0.6 and 0.5 < aspect_ratio < 2.0:
                    mosaic_cells.append((x, y, w, h))

        if len(mosaic_cells) < 30:
            return False

        # --- 決定的な「グリッド性」のチェック ---
        # 近くにあるタイルの粒同士が、同じようなサイズ(w, h)を持っているかを比較
        similar_size_count = 0
        for i in range(len(mosaic_cells)):
            for j in range(i + 1, min(i + 20, len(mosaic_cells))): # 近傍のみ比較
                w1, h1 = mosaic_cells[i][2], mosaic_cells[i][3]
                w2, h2 = mosaic_cells[j][2], mosaic_cells[j][3]
                
                # 幅と高さの差が20%以内なら「同じ規格のタイル」とみなす
                if abs(w1 - w2) < (w1 * 0.2) and abs(h1 - h2) < (h1 * 0.2):
                    similar_size_count += 1
        
        # 自然画ならサイズがバラバラになるが、モザイクなら均一になる
        # 総数に対する「揃っている度合い」で判定
        score = similar_size_count / len(mosaic_cells)
        
        # スコア(密な一致度)としきい値を調整
        # 240pxで判別しにくいものはこの一致度が高くなる傾向があります
        return len(mosaic_cells) > 40 and score > 1.2

    except Exception as e:
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
