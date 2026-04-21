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
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        h, w = img.shape
        if h < 100 or w < 100: return False

        img_resized = cv2.resize(img, (500, 500))
        edges = cv2.Canny(img_resized, 80, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=5)
        
        if lines is None: return False

        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 400: continue
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            
            if angle < 5 or angle > 175:
                horizontal_lines.append(line[0])
            elif 85 < angle < 95:
                vertical_lines.append(line[0])

        if len(horizontal_lines) < 15 or len(vertical_lines) < 15:
            return False

        intersect_count = 0
        for h_line in horizontal_lines:
            hx1, hy1, hx2, hy2 = h_line
            hy_avg = (hy1 + hy2) / 2
            for v_line in vertical_lines:
                vx1, vy1, vx2, vy2 = v_line
                vx_avg = (vx1 + vx2) / 2
                h_x_min, h_x_max = min(hx1, hx2), max(hx1, hx2)
                v_y_min, v_y_max = min(vy1, vy2), max(vy1, vy2)
                if (h_x_min <= vx_avg <= h_x_max) and (v_y_min <= hy_avg <= v_y_max):
                    intersect_count += 1

        return intersect_count > 200

    except Exception as e:
        return False

def main():
    sys.stderr.write("--- Starting Dual-Tagging Scan (Mosaic vs NoMosaic) ---\n")
    
    # 1. 両方のタグIDを取得
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    
    mosaic_id = tags_map.get(TAG_MOSAIC)
    no_mosaic_id = tags_map.get(TAG_NO_MOSAIC)

    if not mosaic_id or not no_mosaic_id:
        sys.stderr.write(f"Error: '{TAG_MOSAIC}' と '{TAG_NO_MOSAIC}' タグを事前に作成してください。\n")
        return

    # 2. 全ID取得
    all_data = stash_query('{ allImages { id } }')
    if not all_data: return
    images = all_data.get('allImages', [])
    total = len(images)
    sys.stderr.write(f"Processing {total} images...\n")

    mosaic_count = 0
    no_mosaic_count = 0

    for count, item in enumerate(images, 1):
        img_id = item['id']
        
        # 3. パス取得
        query = 'query GetPath($id: ID){ findImage(id: $id){ files { path } } }'
        detail = stash_query(query, {"id": img_id})
        if not detail or not detail.get('findImage'): continue
        files = detail['findImage'].get('files', [])
        if not files: continue
        path = files[0].get('path')

        # 4. 判定とタグ付与
        is_m = is_mosaic(path)
        target_tag_id = mosaic_id if is_m else no_mosaic_id
        
        if is_m: mosaic_count += 1
        else: no_mosaic_count += 1

        # タグ付与の実行（GraphQL）
        u_query = 'mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }'
        stash_query(u_query, {"id": img_id, "tags": [target_tag_id]})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} (Mosaic: {mosaic_count}, NoMosaic: {no_mosaic_count})\n")

    sys.stderr.write(f"--- Finished! Mosaic: {mosaic_count}, NoMosaic: {no_mosaic_count} ---\n")

if __name__ == "__main__":
    main()
