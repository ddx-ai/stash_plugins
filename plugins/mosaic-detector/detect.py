import sys
import json
import os
import cv2
import numpy as np
from collections import Counter

# --- Stashライブラリパスの解決 ---
plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(plugin_dir, ".."))

try:
    from stashapi.stashapp import StashInterface
    import stashapi.log as log
except ImportError:
    class FallbackLog:
        def info(self, m): sys.stderr.write(f"INFO: {m}\n")
        def error(self, m): sys.stderr.write(f"ERROR: {m}\n")
    log = FallbackLog()

# --- 入力データの取得 ---
try:
    raw_input = sys.stdin.read()
    json_input = json.loads(raw_input) if raw_input else {}
except:
    json_input = {}

def is_mosaic(path, threshold):
    """Harris Corner 判定ロジック"""
    if not path or not os.path.exists(path): return False, 0, 0, 0
    try:
        img = cv2.imread(path)
        if img is None: return False, 0, 0, 0
        
        target_size = 1024
        img_res = cv2.resize(img, (target_size, target_size))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        _, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        grid_dim = 32
        block_size = target_size // grid_dim
        active_blocks = 0
        for y in range(0, target_size, block_size):
            for x in range(0, target_size, block_size):
                if cv2.countNonZero(dst[y:y+block_size, x:x+block_size]) >= 4:
                    active_blocks += 1
        coverage = round(active_blocks / (grid_dim * grid_dim), 3)

        contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                points.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        
        pts = len(points)
        if not (100 < pts < 5000): return False, coverage, 0, pts

        x_coords = [p[0] for p in points]
        alignment = round(sum(count for val, count in Counter(x_coords).most_common(20)) / pts, 3)
        
        return (coverage >= threshold and alignment > 0.18), coverage, alignment, pts
    except:
        return False, 0, 0, 0

def main():
    if "server_connection" not in json_input:
        sys.stderr.write("ERROR: No server connection info\n")
        return

    # StashInterfaceの初期化 (v0.31.1対応)
    client = StashInterface(json_input["server_connection"])
    log.info("Mosaic Detector: Starting task...")

    # --- 設定取得 (ID: mosaic-detector に準拠) ---
    try:
        full_config = client.get_configuration()
        plugin_config = full_config.get("plugins", {}).get("mosaic-detector", {})
    except:
        plugin_config = {}

    re_check = str(plugin_config.get("ReCheckMode", "false")).lower() == "true"
    try:
        threshold = float(plugin_config.get("Threshold", 0.15))
    except:
        threshold = 0.15

    # --- タグ取得 ---
    m_tag = client.find_tag("Mosaic", create=True)
    n_tag = client.find_tag("NoMosaic", create=True)
    m_id, n_id = str(m_tag["id"]), str(n_tag["id"])

    # --- 高速クエリ (call_GQLを使用) ---
    log.info("Querying database (optimized)...")
    
    if re_check:
        query = "query { allImages { id files { path } tags { id } } }"
        res = client.call_GQL(query)
        targets = res.get('allImages', [])
    else:
        # 未判定のものだけをサーバー側で抽出
        query = """
        query GetUnprocessed($filter: ImageFilterType) {
          allImages(image_filter: $filter) {
            id
            files { path }
            tags { id }
          }
        }
        """
        variables = {
            "filter": {
                "tags": { "value": [m_id, n_id], "modifier": "NOT_IN" }
            }
        }
        res = client.call_GQL(query, variables)
        targets = res.get('allImages', [])

    total = len(targets)
    log.info(f"Target count: {total} images. Starting analysis...")

    # --- 解析ループ ---
    for count, item in enumerate(targets, 1):
        if not item.get("files"): continue
        
        img_id = item["id"]
        path = item["files"][0]["path"]
        current_tids = [str(t["id"]) for t in item.get("tags", [])]

        is_m, cov, alg, pts = is_mosaic(path, threshold)
        new_tag_id = m_id if is_m else n_id
        
        # タグの差し替えロジック
        clean_tags = [tid for tid in current_tids if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_tag_id]

        status = ""
        if set(current_tids) != set(final_tags):
            mutation = """
            mutation Update($id: ID!, $tags: [ID!]) {
              imageUpdate(input: { id: $id, tag_ids: $tags }) { id }
            }
            """
            # ここも修正: call_GQL
            client.call_GQL(mutation, {"id": img_id, "tags": final_tags})
            status = " (Updated)"

        label = "[MOSAIC]" if is_m else "[CLEAN] "
        log.info(f"[{count}/{total}] {label} cov:{cov} alg:{alg} - {os.path.basename(path)}{status}")

    log.info("--- Mosaic Detection Task Completed ---")

if __name__ == "__main__":
    main()
