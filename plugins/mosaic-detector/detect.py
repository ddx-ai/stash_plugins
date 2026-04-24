import sys
import json
import os
import cv2
import numpy as np

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

# --- 入力データ取得 ---
try:
    raw_input = sys.stdin.read()
    json_input = json.loads(raw_input) if raw_input else {}
except:
    json_input = {}

def is_mosaic(path, threshold):
    """
    縦横同期スキャン・ロジック (v3.0)
    縦横の周期が一致することを条件に加え、誤判定を抑制
    """
    if not path or not os.path.exists(path): return False, 0, 0, 0
    try:
        img = cv2.imread(path)
        if img is None: return False, 0, 0, 0
        
        # 比率を維持してリサイズ（正方形に潰さない）
        # これにより、画像内のブロックが「正方形」であることを維持する
        h, w = img.shape[:2]
        target_size = 512
        scale = target_size / max(h, w)
        img_res = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        # エッジ抽出
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mag = np.absolute(laplacian)

        # 投影
        proj_x = np.sum(mag, axis=0)
        proj_y = np.sum(mag, axis=1)

        def get_period_map(proj):
            # 8px〜32pxの各サイズでスコアを算出してリスト化
            p_min, p_max = np.min(proj), np.max(proj)
            if p_max - p_min < 1e-5: return {}
            norm_proj = (proj - p_min) / (p_max - p_min)
            
            period_scores = {}
            for step in range(8, 33):
                # 平均を取ってスコア化
                period_scores[step] = np.mean(norm_proj[::step])
            return period_scores

        scores_x = get_period_map(proj_x)
        scores_y = get_period_map(proj_y)

        if not scores_x or not scores_y: return False, 0, 0, 0

        # --- 縦横同期判定 ---
        max_combined_score = 0
        best_step = 0
        
        for step in range(8, 33):
            # 縦と横のスコアを掛け合わせる（両方高いサイズがあるかを探す）
            # これにより「縦だけ」「横だけ」の周期性は除外される
            combined = scores_x[step] * scores_y[step]
            if combined > max_combined_score:
                max_combined_score = combined
                best_step = step

        # スコアの正規化（0.0〜1.0）
        # sqrtを取ることで、元のスケール感に戻す
        final_score = round(np.sqrt(max_combined_score), 3)
        
        # 判定
        # 同期を条件にすると、自然な画像では数値が極端に低くなるため
        # しきい値は少し下げて 0.3 〜 0.4 あたりが目安になります
        is_m = final_score >= threshold
        
        return is_m, final_score, best_step, 0
    except Exception:
        return False, 0, 0, 0

def main():
    if "server_connection" not in json_input:
        sys.stderr.write("ERROR: No server connection info\n")
        return

    # StashInterface初期化 (v0.31.1対応)
    client = StashInterface(json_input["server_connection"])
    log.info("Mosaic Detector: Starting task (Frequency Analysis Mode)...")

    # --- 設定取得 ---
    try:
        full_config = client.get_configuration()
        # フォルダ名/yml名が mosaic-detector であることを前提
        plugin_config = full_config.get("plugins", {}).get("mosaic-detector", {})
    except:
        plugin_config = {}

    re_check = str(plugin_config.get("ReCheckMode", "false")).lower() == "true"
    try:
        # 新ロジックでは 0.5 程度が標準的なしきい値になります
        threshold = float(plugin_config.get("Threshold", 0.5))
    except:
        threshold = 0.5

    # --- タグ取得 ---
    m_tag = client.find_tag("Mosaic", create=True)
    n_tag = client.find_tag("NoMosaic", create=True)
    m_id, n_id = str(m_tag["id"]), str(n_tag["id"])

    # --- 画像リスト取得 (v0.31.1 互換クエリ) ---
    log.info("Fetching image list from Stash...")
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

    # Python側で未判定分をフィルタリング
    targets = []
    for i in all_images:
        current_tids = [str(t["id"]) for t in i.get("tags", [])]
        if re_check or (m_id not in current_tids and n_id not in current_tids):
            targets.append(i)

    total = len(targets)
    log.info(f"Target count: {total} images. Using Threshold: {threshold}")

    # --- 解析ループ ---
    for count, item in enumerate(targets, 1):
        if not item.get("files"): continue
        
        img_id = item["id"]
        path = item["files"][0]["path"]
        current_tids = [str(t["id"]) for t in item.get("tags", [])]

        # 判定実行
        is_m, score, _, _ = is_mosaic(path, threshold)
        new_tag_id = m_id if is_m else n_id
        
        # 既存判定タグの除去と新タグの追加
        clean_tags = [tid for tid in current_tids if tid not in [m_id, n_id]]
        final_tags = clean_tags + [new_tag_id]

        # 変更がある場合のみ更新
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
        log.info(f"[{count}/{total}] {label} score:{score} - {os.path.basename(path)}{status}")

    log.info("--- Mosaic Detection Task Completed ---")

if __name__ == "__main__":
    main()
