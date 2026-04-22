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

def log_info(message):
    """StashのログにINFOレベルで出力する(JSON形式)"""
    print(json.dumps({"level": "info", "message": message}))
    sys.stdout.flush()

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
    """BulkImageScrape互換の方式で設定を確実に取得する"""
    re_check = False
    threshold = 0.15
    try:
        res = stash_query('{ configuration { plugins } }')
        if not res:
            return False, 0.15
        all_configs = res.get('configuration', {}).get('plugins', {})
        
        config = {}
        for key in all_configs.keys():
            # フォルダ名やプラグイン名に柔軟に対応
            if key.lower() in ["mosaic detector", "mosaic-detector", "stash-plugin"]:
                config = all_configs[key]
                break
        
        if config:
            rc = config.get('ReCheckMode')
            if rc is not None:
                re_check = str(rc).lower() == 'true' if not isinstance(rc, bool) else rc
            tr = config.get('Threshold')
            if tr:
                try: threshold = float(tr)
                except: threshold = 0.15
            return re_check, threshold
    except:
        pass
    return False, 0.15

def is_mosaic(path, threshold):
    """画像解析ロジック"""
    if not path or not os.path.exists(path):
        return False, 0, 0, 0
    try:
        img = cv2.imread(path)
        if img is None: return False, 0, 0, 0
        
        target_size = 1024
        img_res = cv2.resize(img, (target_size, target_size))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        # Harrisコーナー検出
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        _, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # 面積比率（グリッド）
        grid_dim = 32
        block_size = target_size // grid_dim
        active_blocks = 0
        for y in range(0, target_size, block_size):
            for x in range(0, target_size, block_size):
                if cv2.countNonZero(dst[y:y+block_size, x:x+block_size]) >= 4:
                    active_blocks += 1
        coverage = round(active_blocks / (grid_dim * grid_dim), 3)

        # 軸整合性
        contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                points.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        pts = len(points)
        if not (100 < pts < 5000):
            return False, coverage, 0, pts

        x_coords = [p[0] for p in points]
        alignment = round(sum(count for val, count in Counter(x_coords).most_common(20)) / pts, 3)
        
        result = (coverage >= threshold and alignment > 0.18)
        return result, coverage, alignment, pts
    except:
        return False, 0, 0, 0

def main():
    re_check, threshold = get_config()
    log_info(f"--- High-Res Scan Start: {'RE-CHECK' if re_check else 'NEW ONLY'} (Threshold: {threshold}) ---")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    m_id, n_id = tags_map.get(TAG_MOSAIC), tags_map.get(TAG_NO_MOSAIC)

    if not m_id or not n_id:
        log_info("Error: 'Mosaic' and 'NoMosaic' tags must exist.")
        return

    # 2. 画像リスト取得
    res = stash_query('{ allImages { id tags { id } } }')
    all_imgs = res.get('allImages', []) if res else []
    targets = [i for i in all_imgs if re_check or not any(t['id'] in [m_id, n_id] for t in i.get('tags', []))]

    total = len(targets)
    log_info(f"Analyzing {total} images...")

    for count, item in enumerate(targets, 1):
        img_id = item['id']
        detail = stash_query('query G($id: ID){ findImage(id: $id){ files { path } tags { id } } }', {"id": img_id})
        if not detail or not detail['findImage']: continue
        
        img_data = detail['findImage']
        path = img_data['files'][0]['path']
        file_name = os.path.basename(path)
        current_tags = [t['id'] for t in img_data.get('tags', [])]

        # 判定実行
        is_m, cov, alg, pts = is_mosaic(path, threshold)
        new_tag_id = m_id if is_m else n_id
        
        # 旧判定タグを削除して新判定タグを追加
        clean_tags = [tid for tid in current_tags if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_tag_id]

        if set(current_tags) != set(final_tags):
            stash_query('mutation U($id:ID!,$t:[ID!]){ imageUpdate(input:{id:$id,tag_ids:$t}){id} }', 
                        {"id": img_id, "tags": final_tags})
            status = " (Updated)"
        else:
            status = ""

        # INFOログ出力
        label = "[MOSAIC]" if is_m else "[CLEAN] "
        log_info(f"[{count}/{total}] {label} cov:{cov} alg:{alg} pts:{pts} - {file_name}{status}")

    log_info("--- Scan Completed ---")

if __name__ == "__main__":
    main()
