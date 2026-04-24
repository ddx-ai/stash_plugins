import sys
import json
import os
import cv2
import numpy as np
from collections import Counter

# --- パス解決 ---
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

# --- 入力データ取得 ---
try:
    raw_input = sys.stdin.read()
    json_input = json.loads(raw_input) if raw_input else {}
except:
    json_input = {}

def is_mosaic(path, threshold):
    """正方形ブロック構造の周期性解析によるモザイク判定"""
    if not path or not os.path.exists(path): return False, 0, 0, 0
    try:
        img = cv2.imread(path)
        if img is None: return False, 0, 0, 0
        
        # 処理の安定化のためサイズ統一
        target_size = 512 
        img_res = cv2.resize(img, (target_size, target_size))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        # エッジ検出（縦横のラインを強調）
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        mag = np.uint8(np.absolute(mag))

        # 縦横の投影（プロジェクション）による周期性確認
        # モザイクがあれば、特定のピクセル間隔で高いピークが出る
        proj_x = np.sum(mag, axis=0)
        proj_y = np.sum(mag, axis=1)

        def analyze_periodicity(proj):
            # 差分を取ってエッジの立ち上がりを強調
            diff = np.abs(np.diff(proj))
            # 自己相関に似た手法で「一定の間隔でエッジがあるか」をスコア化
            # 8px〜32px程度の正方形ブロックを想定
            max_score = 0
            for block_size in range(8, 32):
                score = 0
                for i in range(0, len(diff) - block_size, block_size):
                    score += diff[i]
                max_score = max(max_score, score)
            return max_score

        score_x = analyze_periodicity(proj_x)
        score_y = analyze_periodicity(proj_y)
        
        # 最終スコアの正規化（画像全体の輝度やコントラストに依存しないよう調整）
        final_score = (score_x + score_y) / (np.mean(gray) + 1) / 1000
        
        # 判定しきい値の調整（0.15前後で調整）
        is_m = final_score > threshold
        
        return is_m, round(final_score, 3), 0, 0
    except Exception as e:
        return False, 0, 0, 0

def main():
    if "server_connection" not in json_input:
        sys.stderr.write("ERROR: No server connection info\n")
        return

    client = StashInterface(json_input["server_connection"])
    log.info("Mosaic Detector: Starting task for Stash v0.31.1...")

    # 設定取得
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

    # タグID取得
    m_tag = client.find_tag("Mosaic", create=True)
    n_tag = client.find_tag("NoMosaic", create=True)
    m_id, n_id = str(m_tag["id"]), str(n_tag["id"])

    # --- v0.31.1 用 互換クエリ ---
    # フィルタを使わず、全件の基本情報を取得（結合を減らしてSQLの奔流を最短化）
    log.info("Fetching image list...")
    query = """
    query {
      allImages {
        id
        files { path }
        tags { id }
      }
    }
    """
    res = client.call_GQL(query)
    all_images = res.get('allImages', [])

    # Python側で高速フィルタリング
    targets = []
    for i in all_images:
        current_tids = [str(t["id"]) for t in i.get("tags", [])]
        if re_check or (m_id not in current_tids and n_id not in current_tids):
            targets.append(i)

    total = len(targets)
    log.info(f"Target count: {total} images. Starting analysis...")

    # 解析ループ
    for count, item in enumerate(targets, 1):
        if not item.get("files"): continue
        
        img_id = item["id"]
        path = item["files"][0]["path"]
        current_tids = [str(t["id"]) for t in item.get("tags", [])]

        is_m, cov, alg, pts = is_mosaic(path, threshold)
        new_tag_id = m_id if is_m else n_id
        
        clean_tags = [tid for tid in current_tids if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_tag_id]

        status = ""
        if set(current_tids) != set(final_tags):
            mutation = """
            mutation Update($id: ID!, $tags: [ID!]) {
              imageUpdate(input: { id: $id, tag_ids: $tags }) { id }
            }
            """
            client.call_GQL(mutation, {"id": img_id, "tags": final_tags})
            status = " (Updated)"

        label = "[MOSAIC]" if is_m else "[CLEAN] "
        log.info(f"[{count}/{total}] {label} {os.path.basename(path)}{status}")

    log.info("--- Mosaic Detection Task Completed ---")

if __name__ == "__main__":
    main()
