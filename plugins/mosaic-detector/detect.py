import requests
import cv2
import numpy as np
import os
import sys
import json
from collections import Counter

# --- 設定とタグ ---
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
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=30)
        return res.json().get('data')
    except:
        return None

def get_config():
    """StashのSettingsから設定を取得。値がない場合はデフォルト値を返す"""
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            plugins = data.get('server_config', {}).get('plugins', {})
            config = plugins.get('Mosaic Detector', {})
            
            re_check = config.get('ReCheckMode', False)
            
            # Thresholdが空または未設定の場合は 0.15 を使う
            threshold_raw = config.get('Threshold')
            threshold = float(threshold_raw) if threshold_raw is not None else 0.15
            
            return re_check, threshold
    except:
        pass
    return False, 0.15

def is_mosaic(path, threshold):
    """面積比率と軸整合性による判定"""
    if not path or not os.path.exists(path):
        return False
    try:
        img = cv2.imread(path)
        if img is None:
            return False
        
        target_size = 1024
        img_res = cv2.resize(img, (target_size, target_size))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        # 1. コーナー検出
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        _, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # 2. 面積比率（グリッドスキャン）
        grid_dim = 32
        block_size = target_size // grid_dim
        active_blocks = 0
        for y in range(0, target_size, block_size):
            for x in range(0, target_size, block_size):
                if cv2.countNonZero(dst[y:y+block_size, x:x+block_size]) >= 4:
                    active_blocks += 1
        
        coverage = active_blocks / (grid_dim * grid_dim)
        
        # UIから取得したしきい値を適用
        if coverage < threshold:
            return False

        # 3. 軸整合性チェック
        contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                points.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        if not (100 < len(points) < 5000):
            return False

        x_coords = [p[0] for p in points]
        # 上位20位のX軸への集中度を計算
        x_alignment = sum(count for val, count in Counter(x_coords).most_common(20)) / len(points)
        
        return x_alignment > 0.18

    except:
        return False

def main():
    re_check_mode, threshold = get_config()
    sys.stderr.write(f"--- Mode: {'RE-CHECK' if re_check_mode else 'NEW ONLY'} (Threshold: {threshold}) ---\n")
    
    # タグID取得
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    m_id, n_id = tags_map.get(TAG_MOSAIC), tags_map.get(TAG_NO_MOSAIC)

    if not m_id or not n_id:
        sys.stderr.write("Error: 'Mosaic' and 'NoMosaic' tags must exist in Stash.\n")
        return

    # 全画像リスト取得
    res = stash_query('{ allImages { id tags { id } } }')
    all_images = res.get('allImages', []) if res else []
    
    targets = []
    for img in all_images:
        tag_ids = [t['id'] for t in img.get('tags', [])]
        # 再判定モードか、未判定（どちらのタグも無い）の場合に対象とする
        if re_check_mode or (m_id not in tag_ids and n_id not in tag_ids):
            targets.append(img)

    total = len(targets)
    sys.stderr.write(f"Analyzing {total} images...\n")

    for count, item in enumerate(targets, 1):
        img_id = item['id']
        detail = stash_query('query G($id: ID){ findImage(id: $id){ files { path } tags { id } } }', {"id": img_id})
        if not detail or not detail['findImage']:
            continue
        
        img_data = detail['findImage']
        path = img_data['files'][0]['path']
        current_tags = [t['id'] for t in img_data.get('tags', [])]

        # 判定実行
        is_m = is_mosaic(path, threshold)
        new_tag_id = m_id if is_m else n_id
        
        # 既存のMosaic/NoMosaicタグを除去して新しいのを追加
        clean_tags = [tid for tid in current_tags if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_tag_id]

        # タグに変更がある場合のみ更新
        if set(current_tags) != set(final_tags):
            stash_query('mutation U($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }', 
                        {"id": img_id, "tags": final_tags})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} images scanned.\n")

    sys.stderr.write("--- Scan Task Completed ---\n")

if __name__ == "__main__":
    main()
