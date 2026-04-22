import requests
import cv2
import numpy as np
import os
import sys
import json

# --- Stashからの引数を受け取る ---
def get_args():
    # Stashは標準入力からJSON形式で引数を渡してきます
    try:
        input_data = sys.stdin.read()
        if input_data:
            return json.loads(input_data).get('args', {})
    except:
        return {}
    return {}

STASH_PORT = os.environ.get("STASH_PORT", "9999")
STASH_URL = f"http://127.0.0.1:{STASH_PORT}/graphql"
TAG_MOSAIC = "Mosaic"
TAG_NO_MOSAIC = "NoMosaic"

# 通信チャネル保護（ログ出力用）
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

def is_mosaic(path):
    if not path or not os.path.exists(path): return False
    try:
        img = cv2.imread(path)
        if img is None: return False
        img_res = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        mag = np.uint8(mag / mag.max() * 255)
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
    # UIからの引数を取得
    args = get_args()
    re_check_mode = args.get('ReCheck', False)
    
    mode_str = "RE-CHECK ALL" if re_check_mode else "NEW IMAGES ONLY"
    sys.stderr.write(f"--- Starting Scan Mode: {mode_str} ---\n")
    
    # タグID取得
    data = stash_query('{ allTags { id name } }')
    tags_map = {t['name']: t['id'] for t in data.get('allTags', [])} if data else {}
    mosaic_id = tags_map.get(TAG_MOSAIC)
    no_mosaic_id = tags_map.get(TAG_NO_MOSAIC)

    if not mosaic_id or not no_mosaic_id:
        sys.stderr.write("Error: Tags not found.\n")
        return

    # モードに応じた画像取得
    if re_check_mode:
        all_data = stash_query('{ allImages { id } }')
        images = all_data.get('allImages', []) if all_data else []
    else:
        q = """
        query GetUnchecked($ids: [ID!]) {
          findImages(image_filter: { tags: { modifier: NOT_INCLUDES, value: $ids } }) {
            images { id }
          }
        }
        """
        unchecked_data = stash_query(q, {"ids": [mosaic_id, no_mosaic_id]})
        images = unchecked_data['findImages']['images'] if unchecked_data else []

    total = len(images)
    sys.stderr.write(f"Target images: {total}\n")

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
        
        clean_tag_ids = [tid for tid in current_tag_ids if tid not in [mosaic_id, no_mosaic_id]]
        updated_tag_ids = clean_tag_ids + [new_tag_id]

        if set(current_tag_ids) != set(updated_tag_ids):
            u_query = 'mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }'
            stash_query(u_query, {"id": img_id, "tags": updated_tag_ids})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} checked.\n")

    sys.stderr.write("--- Scan Finished! ---\n")

if __name__ == "__main__":
    main()
