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
    """Stashが提供する'args'階層から設定を抽出する"""
    re_check = False
    threshold = 0.15
    
    try:
        if not sys.stdin.isatty():
            raw_input = sys.stdin.read()
            if raw_input:
                data = json.loads(raw_input)
                
                # ログから判明した 'args' 階層を探索
                # Stashのタスク実行時は args -> server_config の中にプラグイン設定が入ります
                args = data.get('args', {})
                config = args.get('server_config', {}).get('plugins', {}).get('Mosaic Detector', {})
                
                if not config:
                    # バックアップ：API経由で直接取得を試みる
                    res = stash_query('{ configuration { plugins } }')
                    if res:
                        config = res.get('configuration', {}).get('plugins', {}).get('Mosaic Detector', {})

                if config:
                    sys.stderr.write(f"DEBUG: Found Config: {config}\n")
                    # 大文字小文字を無視してキーを探索
                    for k, v in config.items():
                        if k.lower() == 'recheckmode':
                            re_check = str(v).lower() == 'true'
                        if k.lower() == 'threshold':
                            try:
                                threshold = float(v)
                            except:
                                threshold = 0.15
                    
                    return re_check, threshold
    except Exception as e:
        sys.stderr.write(f"DEBUG: Config Error: {e}\n")

    return re_check, threshold

def is_mosaic(path, threshold):
    """
    面積比率(Coverage)、軸整合性(Alignment)、角の数(Points)を計算
    """
    if not path or not os.path.exists(path):
        return False, 0, 0, 0
    try:
        img = cv2.imread(path)
        if img is None:
            return False, 0, 0, 0
        
        target_size = 1024
        img_res = cv2.resize(img, (target_size, target_size))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        # 1. コーナー検出 (Harris)
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
        
        coverage = round(active_blocks / (grid_dim * grid_dim), 3)

        # 3. 軸整合性チェック
        contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                points.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        pt_count = len(points)
        # 点が少なすぎる、または多すぎる（ノイズ）場合は計算を打ち切り
        if not (100 < pt_count < 5000):
            return False, coverage, 0, pt_count

        x_coords = [p[0] for p in points]
        # 上位20位のX座標への集中度を計算
        alignment = round(sum(count for val, count in Counter(x_coords).most_common(20)) / pt_count, 3)
        
        # 最終判定：面積がしきい値以上、かつ軸が一定以上揃っていること
        result = (coverage >= threshold and alignment > 0.18)
        return result, coverage, alignment, pt_count

    except:
        return False, 0, 0, 0

def main():
    re_check_mode, threshold = get_config()
    sys.stderr.write(f"--- High-Res Scan Mode: {'RE-CHECK' if re_check_mode else 'NEW ONLY'} (Threshold: {threshold}) ---\n")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    m_id, n_id = tags_map.get(TAG_MOSAIC), tags_map.get(TAG_NO_MOSAIC)

    if not m_id or not n_id:
        sys.stderr.write(f"Error: '{TAG_MOSAIC}' or '{TAG_NO_MOSAIC}' tags missing in Stash.\n")
        return

    # 2. 全画像リスト取得
    res = stash_query('{ allImages { id tags { id } } }')
    all_images = res.get('allImages', []) if res else []
    
    targets = []
    for img in all_images:
        tag_ids = [t['id'] for t in img.get('tags', [])]
        if re_check_mode or (m_id not in tag_ids and n_id not in tag_ids):
            targets.append(img)

    total = len(targets)
    sys.stderr.write(f"Target images: {total}\n")

    # 3. 解析ループ
    for count, item in enumerate(targets, 1):
        img_id = item['id']
        detail = stash_query('query G($id: ID){ findImage(id: $id){ files { path } tags { id } } }', {"id": img_id})
        if not detail or not detail['findImage']:
            continue
        
        img_data = detail['findImage']
        path = img_data['files'][0]['path']
        file_name = os.path.basename(path)
        current_tags = [t['id'] for t in img_data.get('tags', [])]

        # 判定実行（詳細数値を取得）
        is_m, cov, alg, pts = is_mosaic(path, threshold)
        
        new_tag_id = m_id if is_m else n_id
        result_label = "[MOSAIC]" if is_m else "[CLEAN] "
        
        # 判定詳細をログに出力
        sys.stderr.write(f"[{count}/{total}] {result_label} cov:{cov:.3f} alg:{alg:.3f} pts:{pts} - {file_name}\n")

        # タグ更新処理
        clean_tags = [tid for tid in current_tags if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_tag_id]

        if set(current_tags) != set(final_tags):
            stash_query('mutation U($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }', 
                        {"id": img_id, "tags": final_tags})

    sys.stderr.write("--- All Scan Tasks Completed ---\n")

if __name__ == "__main__":
    main()
