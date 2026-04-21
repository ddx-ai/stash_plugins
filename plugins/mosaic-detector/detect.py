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
TARGET_TAG = "Mosaic"

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
    """
    高度なモザイク検出ロジック
    4コマ漫画、ドット絵、床、縦じまを除外する。
    """
    if not path or not os.path.exists(path): return False
    try:
        # 1. 画像の読み込みと前処理（解析用に少し大きく設定）
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return False
        h, w = img.shape
        if h < 100 or w < 100: return False # 小さすぎる画像は除外

        # 解析サイズを統一（500x500）
        img_resized = cv2.resize(img, (500, 500))
        
        # 2. エッジ検出（少し閾値を厳しく）
        edges = cv2.Canny(img_resized, 80, 200)

        # 3. 確率的Hough線変換
        # 閾値を調整し、細かすぎる線やノイズを排除
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=5)
        
        if lines is None: return False

        horizontal_lines = []
        vertical_lines = []
        
        # 線の選別（4コマ漫画の長い線や、床の斜め線を排除）
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # あまりに長い線（> 400px）は4コマのコマ割りや床の可能性が高いので除外
            if length > 400: continue
            
            # 角度の計算
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            
            # 水平線 (0度付近)
            if angle < 5 or angle > 175:
                horizontal_lines.append(line[0])
            # 垂直線 (90度付近)
            elif 85 < angle < 95:
                vertical_lines.append(line[0])

        # 4. 「格子（交差点）」の検出（ここが最重要）
        # 単なる平行線（縦じま、床）ではなく、縦横が交わっているかを見る
        if len(horizontal_lines) < 15 or len(vertical_lines) < 15:
            return False # 縦横どちらかが少ない場合は格子ではない（縦じま・床を除外）

        intersect_count = 0
        
        # すべての水平線と垂直線の交差をチェック（力技だが500x500なら速い）
        for h_line in horizontal_lines:
            hx1, hy1, hx2, hy2 = h_line
            # 水平線のY座標（平均）
            hy_avg = (hy1 + hy2) / 2
            
            for v_line in vertical_lines:
                vx1, vy1, vx2, vy2 = v_line
                # 垂直線のX座標（平均）
                vx_avg = (vx1 + vx2) / 2
                
                # 交差判定（水平線のX範囲に垂直線のXがあり、垂直線のY範囲に水平線のYがあるか）
                h_x_min, h_x_max = min(hx1, hx2), max(hx1, hx2)
                v_y_min, v_y_max = min(vy1, vy2), max(vy1, vy2)
                
                if (h_x_min <= vx_avg <= h_x_max) and (v_y_min <= hy_avg <= v_y_max):
                    intersect_count += 1

        # 5. 判定（交差点＝格子の数が一定以上あるか）
        # ドット絵は格子が多いが、線が短すぎるためHoughLinesPで弾かれる。
        # 4コマ漫画は格子が少なすぎる。
        # 床・縦じまは交差点がほぼゼロ。
        # 200個以上の交差点があれば、それは「モザイクの格子」である可能性が極めて高い。
        return intersect_count > 200

    exceptException as e:
        sys.stderr.write(f"Error analyzing {os.path.basename(path)}: {e}\n")
        return False

def main():
    sys.stderr.write("--- Starting Advanced Mosaic Detector (Anti-False-Positive) ---\n")
    
    # タグID取得
    data = stash_query('{ allTags { id name } }')
    tag_id = next((t['id'] for t in data.get('allTags', []) if t['name'] == TARGET_TAG), None) if data else None
    if not tag_id:
        sys.stderr.write(f"Error: タグ '{TARGET_TAG}' を作成してください。\n")
        return

    # 全ID取得
    all_data = stash_query('{ allImages { id } }')
    if not all_data: return
    images = all_data.get('allImages', [])
    total = len(images)
    sys.stderr.write(f"Total {total} images. Starting advanced scan...\n")

    detected_count = 0
    for count, item in enumerate(images, 1):
        img_id = item['id']
        
        # パス取得 (files { path })
        query = 'query GetPath($id: ID){ findImage(id: $id){ files { path } } }'
        detail = stash_query(query, {"id": img_id})
        
        if not detail or not detail.get('findImage'): continue
        files = detail['findImage'].get('files', [])
        if not files: continue
        path = files[0].get('path')

        # 強化された解析
        if is_mosaic(path):
            detected_count += 1
            sys.stderr.write(f"[{count}/{total}] Mosaic! -> {os.path.basename(path)}\n")
            
            # タグ付与
            u_query = 'mutation($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }'
            stash_query(u_query, {"id": img_id, "tags": [tag_id]})

        if count % 100 == 0:
            sys.stderr.write(f"Progress: {count}/{total} checked...\n")

    sys.stderr.write(f"--- Finished! Found: {detected_count} ---\n")

if __name__ == "__main__":
    main()
