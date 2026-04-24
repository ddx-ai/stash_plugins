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
    正方形ブロックの周期性を解析するロジック (v2.0)
    オーバーフロー対策済・正規化スコア算出
    """
    if not path or not os.path.exists(path): return False, 0, 0, 0
    try:
        # 画像読み込み
        img = cv2.imread(path)
        if img is None: return False, 0, 0, 0
        
        # 解析サイズを512pxに固定（処理速度と精度のバランス）
        target_size = 1024
        img_res = cv2.resize(img, (target_size, target_size))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        # 1. 鮮鋭度の高いエッジ（格子状の線）を抽出
        # CV_64Fを使用して計算中のオーバーフローを防止
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mag = np.absolute(laplacian)

        # 2. 縦横の投影（プロジェクション）
        # 軸方向に合計を取り、エッジの分布を調べる
        proj_x = np.sum(mag, axis=0)
        proj_y = np.sum(mag, axis=1)

        def get_periodicity_score(proj):
            # データの正規化（画像の明るさの影響を排除）
            p_min, p_max = np.min(proj), np.max(proj)
            if p_max - p_min < 1e-5: return 0
            norm_proj = (proj - p_min) / (p_max - p_min)

            # 周期性のスキャン（ブロックサイズ 8px 〜 32px を想定）
            best_score = 0
            for step in range(8, 33):
                # 一定間隔(step)ごとの強度の平均を計算（オーバーフロー防止）
                score = np.mean(norm_proj[::step])
                if score > best_score:
                    best_score = score
            return best_score

        score_x = get_periodicity_score(proj_x)
        score_y = get_periodicity_score(proj_y)

        # 3. 最終判定スコア算出 (0.0 〜 1.0)
        # 縦横両方の規則性を合成
        final_score = round((score_x + score_y) / 2, 3)
        
        # 判定 (デフォルトしきい値目安: 0.5)
        is_m = final_score >= threshold
        
        return is_m, final_score, 0, 0
    except Exception as e:
        # 予期せぬエラー時はログを出してスキップ
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
