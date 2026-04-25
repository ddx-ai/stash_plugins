import sys
import json
import os
import cv2
import numpy as np
import time
from datetime import datetime

# --- タイムゾーン設定 (JST) ---
os.environ['TZ'] = 'Asia/Tokyo'
if hasattr(time, 'tzset'):
    time.tzset()

# --- Stashライブラリパス解決 ---
plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(plugin_dir, ".."))

try:
    from stashapi.stashapp import StashInterface
    import stashapi.log as log
except ImportError:
    class FallbackLog:
        def info(self, m):
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sys.stderr.write(f"INFO[{now}] {m}\n")
        def error(self, m):
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sys.stderr.write(f"ERRO[{now}] {m}\n")
    log = FallbackLog()

def is_mosaic(path, tolerance=15):
    """
    勾配角度フィルタリングを用いた周波数解析 (v4.0)
    斜線（鉛筆画等）を排除し、正方形グリッドのみをスコア化
    """
    if not path or not os.path.exists(path): return 0, 0
    try:
        img = cv2.imread(path)
        if img is None: return 0, 0
        
        h, w = img.shape[:2]
        target_size = 512
        scale = target_size / max(h, w)
        img_res = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        # 勾配算出
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(np.abs(sobel_y), np.abs(sobel_x)) * 180 / np.pi
        mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # 角度フィルタ：0度・90度付近以外のエッジ（斜線）をカット
        mask = ((angles < tolerance) | (angles > 90 - tolerance))
        mag_filtered = np.where(mask, mag, 0)

        proj_x = np.sum(mag_filtered, axis=0)
        proj_y = np.sum(mag_filtered, axis=1)

        def get_period_map(proj):
            p_min, p_max = np.min(proj), np.max(proj)
            if p_max - p_min < 1e-5: return {}
            norm_proj = (proj - p_min) / (p_max - p_min)
            return {step: np.mean(norm_proj[::step]) for step in range(8, 33)}

        scores_x = get_period_map(proj_x)
        scores_y = get_period_map(proj_y)

        if not scores_x or not scores_y: return 0, 0

        max_combined = 0
        best_step = 0
        for step in range(8, 33):
            combined = scores_x[step] * scores_y[step]
            if combined > max_combined:
                max_combined = combined
                best_step = step
        
        return round(np.sqrt(max_combined), 3), best_step
    except Exception:
        return 0, 0

def get_mosaic_tag_name(score, min_threshold):
    """スコア 0.1〜1.0 を 9段階にマッピング"""
    if score < min_threshold: return "NoMosaic"
    idx = int((score - 0.1) / 0.1) + 1
    idx = min(max(idx, 1), 9)
    return f"Mosaic_{idx:02d}"

def main():
    try:
        input_data = json.loads(sys.stdin.read())
    except Exception:
        return

    server = input_data.get("server_connection", {})
    if not server: return
    client = StashInterface(server)

    # UI設定のパース
    args = input_data.get("args", {})
    def get_arg(key, default, type_func):
        val = args.get(key)
        if val is None or val == "": return default
        try:
            if type_func == bool:
                return str(val).lower() in ("true", "1", "yes", "on")
            return type_func(val)
        except:
            return default

    re_check = get_arg("ReCheckMode", False, bool)
    angle_tol = get_arg("AngleTolerance", 15, int)
    min_score = get_arg("ThresholdMin", 0.1, float)
    target_tag_name = str(args.get("TargetTag", "")).strip()

    log.info(f"Mosaic Detector v1.9 [JST] starting... (Tol:{angle_tol}, Min:{min_score})")

    # タグ準備
    managed_names = ["NoMosaic"] + [f"Mosaic_{i:02d}" for i in range(1, 10)]
    tag_map = {name: str(client.find_tag(name, create=True)["id"]) for name in managed_names}
    managed_ids = list(tag_map.values())

    # 画像取得
    res = client.call_GQL("query { allImages { id files { path } tags { id name } } }")
    all_images = res.get('allImages', [])

    targets = []
    for i in all_images:
        c_tags = i.get("tags", [])
        c_tids = [str(t["id"]) for t in c_tags]
        c_names = [t["name"] for t in c_tags]

        if target_tag_name and target_tag_name not in c_names: continue
        if re_check or not any(tid in managed_ids for tid in c_tids):
            targets.append(i)

    total = len(targets)
    log.info(f"Analysis Target: {total} images.")

    for count, item in enumerate(targets, 1):
        if not item.get("files"): continue
        path = item["files"][0]["path"]
        
        score, step = is_mosaic(path, tolerance=angle_tol)
        tag_name = get_mosaic_tag_name(score, min_score)
        new_tag_id = tag_map[tag_name]

        current_tids = [str(t["id"]) for t in item.get("tags", [])]
        
        if new_tag_id not in current_tids:
            final_tags = [tid for tid in current_tids if tid not in managed_ids] + [new_tag_id]
            client.call_GQL(
                "mutation Update($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }",
                {"id": item["id"], "tags": final_tags}
            )

        # 10件ごと、または高スコア時にJSTログ出力
        if count % 10 == 0 or score >= 0.5:
             log.info(f"[{count}/{total}] {score:.3f} ({step}px) -> {tag_name} | {os.path.basename(path)}")

    log.info("--- Mosaic Detection Task Successfully Completed ---")

if __name__ == "__main__":
    main()
