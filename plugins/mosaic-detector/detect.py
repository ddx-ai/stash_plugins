import sys
import json
import os
import cv2
import numpy as np
from collections import Counter

# --- stashapiの確実なインポート ---
try:
    from PythonDepManager import ensure_import
    ensure_import("stashapi")
    import stashapi.log as log
    from stashapi.stash_types import StashItem
    from stashapi.stashapp import StashInterface
except ImportError:
    print(json.dumps({"level": "error", "message": "stashapi not found"}))
    sys.exit(1)

# --- 設定値 ---
TAG_MOSAIC = "Mosaic"
TAG_NO_MOSAIC = "NoMosaic"

def is_mosaic(path, threshold):
    """高解像度 Harris Corner 判定ロジック"""
    if not path or not os.path.exists(path):
        return False, 0, 0, 0
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

        pt_count = len(points)
        if not (100 < pt_count < 5000):
            return False, coverage, 0, pt_count

        x_coords = [p[0] for p in points]
        alignment = round(sum(count for val, count in Counter(x_coords).most_common(20)) / pt_count, 3)
        
        return (coverage >= threshold and alignment > 0.18), coverage, alignment, pt_count
    except Exception as e:
        log.debug(f"Analysis error: {e}")
        return False, 0, 0, 0

def main():
    # stdinから接続情報を取得
    json_input = json.loads(sys.stdin.read())
    client = StashInterface(json_input["server_connection"])
    
    log.info("Mosaic Detector: Starting task...")

    # 設定取得
    plugin_config = client.get_configuration().get("plugins", {}).get("Mosaic Detector", {})
    re_check = str(plugin_config.get("ReCheckMode", "false")).lower() == "true"
    try:
        threshold = float(plugin_config.get("Threshold", 0.15))
    except:
        threshold = 0.15

    log.info(f"Config: Mode={'RE-CHECK' if re_check else 'NEW ONLY'}, Threshold={threshold}")

    # タグID取得 (StashItem.TAG を使用して明示的に取得)
    m_tag = client.find_tag(TAG_MOSAIC, create=True)
    n_tag = client.find_tag(TAG_NO_MOSAIC, create=True)
    m_id, n_id = str(m_tag["id"]), str(n_tag["id"])

    # 全画像取得 (StashItem.IMAGE を指定)
    log.info("Fetching images from Stash...")
    all_images = client.find_images(filter={"per_page": -1}, get_count=False)
    
    targets = []
    for img in all_images:
        current_tids = [str(t["id"]) for t in img.get("tags", [])]
        if re_check or (m_id not in current_tids and n_id not in current_tids):
            targets.append(img)

    total = len(targets)
    log.info(f"Target count: {total} images.")

    # 解析ループ
    for count, item in enumerate(targets, 1):
        img_id = item["id"]
        img_detail = client.find_image(img_id)
        if not img_detail or not img_detail.get("files"): continue
            
        path = img_detail["files"][0]["path"]
        file_name = os.path.basename(path)
        current_tids = [str(t["id"]) for t in img_detail.get("tags", [])]

        is_m, cov, alg, pts = is_mosaic(path, threshold)
        new_tag_id = m_id if is_m else n_id
        
        # タグの完全洗浄と差し替え
        clean_tags = [tid for tid in current_tids if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_tag_id]

        status_label = ""
        # 文字列比較で確実な更新判定
        if set(current_tids) != set(final_tags):
            # 確実なGraphQL呼び出し
            mutation = """
            mutation Update($id: ID!, $tags: [ID!]) {
                imageUpdate(input: { id: $id, tag_ids: $tags }) { id }
            }
            """
            client.call_gql(mutation, {"id": img_id, "tags": final_tags})
            status_label = " (Updated)"

        result_text = "[MOSAIC]" if is_m else "[CLEAN] "
        log.info(f"[{count}/{total}] {result_text} cov:{cov} alg:{alg} pts:{pts} - {file_name}{status_label}")

    log.info("--- Mosaic Detection Task Completed ---")

if __name__ == "__main__":
    main()
