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
            # server_config -> plugins -> Mosaic Detector -> ReCheckMode
            plugins = data.get('server_config', {}).get('plugins', {})
            # フォルダ名ではなく、ymlの 'name' フィールドで指定した名前がキーになります
            config = plugins.get('Mosaic Detector', {})
            return config.get('ReCheckMode', False)
    except:
        pass
    return False

def is_mosaic(path):
    """タイル密度解析"""
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path)
        if img is None: return False
        img_res = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        mag = np.uint8(mag / (mag.max() if mag.max() > 0 else 1) * 255)
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(mag, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mosaic_candidate_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 40 < area < 3000:
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if 4 <= len(approx) <= 12:
                    mosaic_candidate_count += 1
        return mosaic_candidate_count > 35
    except:
        return False

def main():
    re_check_mode = get_config()
    sys.stderr.write(f"--- Mode: {'RE-CHECK' if re_check_mode else 'NEW ONLY'} ---\n")
    
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    m_id, n_id = tags_map.get(TAG_MOSAIC), tags_map.get(TAG_NO_MOSAIC)

    if not m_id or not n_id:
        sys.stderr.write("Error: Mosaic/NoMosaic tags missing.\n")
        return

    if re_check_mode:
        res = stash_query('{ allImages { id } }')
        images = res.get('allImages', []) if res else []
    else:
        q = 'query U($ids: [ID!]) { findImages(image_filter: { tags: { modifier: NOT_INCLUDES, value: $ids } }) { images { id } } }'
        res = stash_query(q, {"ids": [m_id, n_id]})
        images = res['findImages']['images'] if res else []

    total = len(images)
    sys.stderr.write(f"Target: {total} images.\n")

    for count, item in enumerate(images, 1):
        img_id = item['id']
        detail = stash_query('query G($id: ID){ findImage(id: $id){ files { path } tags { id } } }', {"id": img_id})
        if not detail or not detail['findImage']: continue
        img_data = detail['findImage']
        path = img_data['files'][0]['path']
        current_tag_ids = [t['id'] for t in img_data.get('tags', [])]
        
        is_m = is_mosaic(path)
        new_tag_id = m_id if is_m else n_id
        updated_tag_ids = [tid for tid in current_tag_ids if tid not in [m_id, n_id]] + [new_tag_id]

        if set(current_tag_ids) != set(updated_tag_ids):
            stash_query('mutation U($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }', 
                        {"id": img_id, "tags": updated_tag_ids})
        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total}\n")

    sys.stderr.write("--- Done ---\n")

if __name__ == "__main__":
    main()
