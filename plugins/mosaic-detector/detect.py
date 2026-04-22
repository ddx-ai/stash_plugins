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
    if variables: payload['variables'] = variables
    try:
        res = requests.post(STASH_URL, json=payload, headers=headers, timeout=30)
        return res.json().get('data')
    except:
        return None

def get_config():
    """StashのSettingsから設定を取得。失敗時はFalseを返す"""
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            plugins = data.get('server_config', {}).get('plugins', {})
            config = plugins.get('Mosaic Detector', {})
            return config.get('ReCheckMode', False)
    except:
        pass
    return False

def is_mosaic(path):
    """
    軸整合性スキャン：
    角の距離だけでなく、それらが水平・垂直のライン(グリッド)上に
    正しく並んでいるかを厳格に判定し、自然画の誤検知を排除します。
    """
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path)
        if img is None: return False
        
        h, w = img.shape[:2]
        target_size = 1024
        img_res = cv2.resize(img, (target_size, target_size))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        # 1. コーナー検出
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                points.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        
        if len(points) < 100 or len(points) > 5000: # 点が多すぎる(ノイズ)場合も除外
            return False

        # 2. 軸の整合性チェック (水平・垂直に並んでいるか)
        # モザイクの角は、同じX座標または同じY座標に集中する性質があります
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # 座標の出現頻度をカウント
        x_counts = Counter(x_coords).most_common(20)
        y_counts = Counter(y_coords).most_common(20)
        
        # 上位20個の軸に、全ポイントの何割が乗っているか
        # 普通の画像はバラバラですが、モザイクは特定のラインに集中します
        x_alignment = sum(count for val, count in x_counts) / len(points)
        y_alignment = sum(count for val, count in y_counts) / len(points)

        # 3. 隣接距離の規則性
        distances = []
        sample_points = points[:400]
        for i in range(len(sample_points)):
            p1 = sample_points[i]
            for j in range(i + 1, min(i + 20, len(sample_points))):
                p2 = sample_points[j]
                dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                if 8 < dist < 50:
                    distances.append(round(dist, 0))

        if not distances: return False
        dist_counts = Counter(distances)
        common_dist, dist_freq = dist_counts.most_common(1)[0]
        
        # --- 判定しきい値 ---
        # 軸の一致度 (alignment) が高く、かつ特定の距離が支配的であること
        # 普通の画像(風景など)はこの x_alignment が 0.1 以下になることが多いです
        is_aligned = (x_alignment > 0.15 or y_alignment > 0.15)
        has_regular_dist = (dist_freq > 35)

        return is_aligned and has_regular_dist

    except:
        return False

def main():
    re_check_mode = get_config()
    sys.stderr.write(f"--- High-Res Scan Mode: {'RE-CHECK' if re_check_mode else 'NEW ONLY'} ---\n")
    
    # 1. タグID取得
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    m_id, n_id = tags_map.get(TAG_MOSAIC), tags_map.get(TAG_NO_MOSAIC)

    if not m_id or not n_id:
        sys.stderr.write(f"Error: Tags '{TAG_MOSAIC}' and '{TAG_NO_MOSAIC}' must exist.\n")
        return

    # 2. 全画像を取得してPython側で安全にフィルタリング
    sys.stderr.write("Fetching image list...\n")
    res = stash_query('{ allImages { id tags { id } } }')
    all_images = res.get('allImages', []) if res else []
    
    targets = []
    if re_check_mode:
        targets = all_images
    else:
        for img in all_images:
            tag_ids = [t['id'] for t in img.get('tags', [])]
            if m_id not in tag_ids and n_id not in tag_ids:
                targets.append(img)

    total = len(targets)
    sys.stderr.write(f"Analyzing {total} images...\n")

    # 3. 解析ループ
    for count, item in enumerate(targets, 1):
        img_id = item['id']
        detail = stash_query('query G($id: ID){ findImage(id: $id){ files { path } tags { id } } }', {"id": img_id})
        if not detail or not detail['findImage']: continue
        
        img_data = detail['findImage']
        path = img_data['files'][0]['path']
        current_tags = [t['id'] for t in img_data.get('tags', [])]

        # 判定実行
        is_m = is_mosaic(path)
        new_id = m_id if is_m else n_id
        
        # 既存のMosaic/NoMosaicタグを入れ替える
        clean_tags = [tid for tid in current_tags if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_id]

        if set(current_tags) != set(final_tags):
            stash_query('mutation U($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }', 
                        {"id": img_id, "tags": final_tags})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} done...\n")

    sys.stderr.write("--- Task Finished ---\n")

if __name__ == "__main__":
    main()
