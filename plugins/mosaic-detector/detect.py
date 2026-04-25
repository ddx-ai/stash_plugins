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

# --- Path setup ---
plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(plugin_dir, ".."))

try:
    from stashapi.stashapp import StashInterface
    import stashapi.log as log
except ImportError:
    class FallbackLog:
        def info(self, m): sys.stderr.write(f"INFO {m}\n")
        def error(self, m): sys.stderr.write(f"ERRO {m}\n")
    log = FallbackLog()

def is_mosaic(path, tolerance=15):
    if not path or not os.path.exists(path): return 0, 0
    try:
        img = cv2.imread(path)
        if img is None: return 0, 0
        h, w = img.shape[:2]
        target_size = 512
        scale = target_size / max(h, w)
        img_res = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(np.abs(sobel_y), np.abs(sobel_x)) * 180 / np.pi
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
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
                max_combined, best_step = combined, step
        return round(np.sqrt(max_combined), 3), best_step
    except: return 0, 0

def get_mosaic_tag_name(score, min_threshold):
    if score < min_threshold: return "NoMosaic"
    idx = int((score - 0.1) / 0.1) + 1
    return f"Mosaic_{min(max(idx, 1), 9):02d}"

def main():
    try:
        input_data = json.loads(sys.stdin.read())
    except: return
    client = StashInterface(input_data.get("server_connection", {}))
    args = input_data.get("args", {})

    try:
        config_all = client.get_configuration().get("plugins", {})
        plugin_settings = config_all.get("mosaic-detector", {})
    except: plugin_settings = {}

    def get_val(key, default, type_func):
        val = plugin_settings.get(key)
        if val is None or (isinstance(val, str) and val == ""): return default
        try:
            if type_func == bool: return str(val).lower() in ("true", "1", "yes", "on")
            return type_func(val)
        except: return default

    re_check = get_val("ReCheckMode", False, bool)
    angle_tol = get_val("AngleTolerance", 15, int)
    min_score = get_val("ThresholdMin", 0.1, float)
    target_tag = str(plugin_settings.get("TargetTag", "")).strip()

    log.info(f"--- Config: ReCheck={re_check}, Angle={angle_tol}, Min={min_score} ---")

    managed_names = ["NoMosaic"] + [f"Mosaic_{i:02d}" for i in range(1, 10)]
    tag_map = {name: str(client.find_tag(name, create=True)["id"]) for name in managed_names}
    managed_ids = list(tag_map.values())

    all_images = []
    cursor, page_size = 1, 2000
    tag_filter = ""
    if target_tag:
        t_tag = client.find_tag(target_tag)
        if t_tag: tag_filter = f', image_filter: {{ tags: {{ value: ["{t_tag["id"]}"], modifier: INCLUDES }} }}'

    log.info("Loading metadata...")
    while True:
        query = "query FindImages($p: Int, $pp: Int) { findImages(filter: { page: $p, per_page: $pp } %s) { count images { id files { path } tags { id } } } }" % tag_filter
        result = client.call_GQL(query, {"p": cursor, "pp": page_size})
        data = result.get("findImages", {})
        images = data.get("images", [])
        if not images: break
        all_images.extend(images)
        if len(all_images) >= data.get("count", 0): break
        cursor += 1

    targets = all_images if re_check else [i for i in all_images if not any(str(t["id"]) in managed_ids for t in i.get("tags", []))]
    total = len(targets)
    log.info(f"Target: {total} images.")

    for count, item in enumerate(targets, 1):
        if not item.get("files"): continue
        score, step = is_mosaic(item["files"][0]["path"], tolerance=angle_tol)
        tag_name = get_mosaic_tag_name(score, min_score)
        new_tag_id = tag_map[tag_name]
        current_tids = [str(t["id"]) for t in item.get("tags", [])]
        if new_tag_id not in current_tids:
            final_tags = [tid for tid in current_tids if tid not in managed_ids] + [new_tag_id]
            client.call_GQL("mutation Update($id: ID!, $tags: [ID!]) { imageUpdate(input: { id: $id, tag_ids: $tags }) { id } }", {"id": item["id"], "tags": final_tags})
        if count % 20 == 0 or score >= 0.5:
            log.info(f"[{count}/{total}] {score:.3f} -> {tag_name}")

if __name__ == "__main__":
    main()
