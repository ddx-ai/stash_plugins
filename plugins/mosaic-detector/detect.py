import sys
import json
import os
import cv2
import numpy as np
from collections import Counter

# --- Stashのライブラリパスを強制的に追加 ---
# Stashの実行ファイルがある場所の周辺からライブラリを探す
plugin_dir = os.path.dirname(os.path.realpath(__file__))
# 通常、stash-python-sdkやstashapiはプラグインディレクトリの親や、特定の場所にあります
# ここではStashが提供する標準的な場所を推測して追加します
sys.path.append(os.path.join(plugin_dir, ".."))

try:
    import stashapi.log as log
    from stashapi.stash_types import StashItem
    from stashapi.stashapp import StashInterface
except ImportError:
    # 直接インポートできない場合は、さらにパスを探るか、エラーを表示
    sys.stderr.write("DEBUG: Standard import failed, attempting fallback...\n")
    # ここで落ちると exit 1 になるので、最低限のロガーを自作
    class FallbackLog:
        def info(self, m): sys.stderr.write(f"INFO: {m}\n")
        def error(self, m): sys.stderr.write(f"ERROR: {m}\n")
    log = FallbackLog()

# --- 入力データの取得 ---
try:
    raw_input = sys.stdin.read()
    json_input = json.loads(raw_input)
except:
    json_input = {}

# --- 解析ロジック ---
def is_mosaic(path, threshold):
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

    # StashInterfaceの初期化
    try:
        client = StashInterface(json_input["server_connection"])
    except:
        sys.stderr.write("ERROR: Could not initialize StashInterface\n")
        return

    log.info("Mosaic Detector: Starting task...")

    # 設定取得
    # client.get_configuration() が使えない場合を想定して安全に取得
    config = json_input.get("args", {}).get("defaults", {}) # fallback
    try:
        full_config = client.get_configuration()
        plugin_config = full_config.get("plugins", {}).get("Mosaic Detector", {})
    except:
        plugin_config = {}

    re_check = str(plugin_config.get("ReCheckMode", "false")).lower() == "true"
    try:
        threshold = float(plugin_config.get("Threshold", 0.15))
    except:
        threshold = 0.15

    log.info(f"Config: Mode={'RE-CHECK' if re_check else 'NEW ONLY'}, Threshold={threshold}")

    # タグ取得
    m_tag = client.find_tag("Mosaic", create=True)
    n_tag = client.find_tag("NoMosaic", create=True)
    m_id, n_id = str(m_tag["id"]), str(n_tag["id"])

    # 全画像取得
    log.info("Fetching images...")
    all_images = client.find_images(filter={"per_page": -1}, get_count=False)
    
    targets = [i for i in all_images if re_check or not any(str(t["id"]) in [m_id, n_id] for t in i.get("tags", []))]
    log.info(f"Analyzing {len(targets)} images.")

    for count, item in enumerate(targets, 1):
        img_id = item["id"]
        # 詳細取得
        img_detail = client.find_image(img_id)
        if not img_detail or not img_detail.get("files"): continue
        path = img_detail["files"][0]["path"]
        current_tids = [str(t["id"]) for t in img_detail.get("tags", [])]

        is_m, cov, alg, pts = is_mosaic(path, threshold)
        new_tag_id = m_id if is_m else n_id
        
        # タグ更新
        clean_tags = [tid for tid in current_tids if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_tag_id]

        if set(current_tids) != set(final_tags):
            mutation = """
            mutation Update($id: ID!, $tags: [ID!]) {
                imageUpdate(input: { id: $id, tag_ids: $tags }) { id }
            }
            """
            client.call_gql(mutation, {"id": img_id, "tags": final_tags})
            updated = " (Updated)"
        else:
            updated = ""

        label = "[MOSAIC]" if is_m else "[CLEAN] "
        log.info(f"[{count}/{len(targets)}] {label} {os.path.basename(path)}{updated}")

if __name__ == "__main__":
    main()
