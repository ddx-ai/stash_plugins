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

def is_mosaic(path):
    """
    勾配角度フィルタリング付き・縦横同期スキャン (v4.0)
    斜線（鉛筆画・ハッチング）を排除し、純粋な水平・垂直グリッドのみを抽出
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

        # 1. 勾配の強度と「角度」を算出
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 角度(radian)を計算
        angles = np.arctan2(np.abs(sobel_y), np.abs(sobel_x)) * 180 / np.pi
        mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # 2. 角度によるフィルタリング
        # 垂直(90度付近)または水平(0度付近)のエッジだけを残す
        # 鉛筆画の斜線（30〜60度付近）はここでゼロになる
        tolerance = 15 # 許容誤差±15度
        mask = ((angles < tolerance) | (angles > 90 - tolerance))
        mag_filtered = np.where(mask, mag, 0)

        # 3. 投影（フィルタリング後の強度を使用）
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

        max_combined_score = 0
        best_step = 0
        for step in range(8, 33):
            combined = scores_x[step] * scores_y[step]
            if combined > max_combined_score:
                max_combined_score = combined
                best_step = step

        final_score = round(np.sqrt(max_combined_score), 3)
        return final_score, best_step
    except:
        return 0, 0

def get_mosaic_tag_name(score):
    """スコア 0.0〜1.0 を 01〜09 のタグに細分化"""
    if score < 0.1: return "NoMosaic"
    if score < 0.2: return "Mosaic_01"
    if score < 0.3: return "Mosaic_02"
    if score < 0.4: return "Mosaic_03"
    if score < 0.5: return "Mosaic_04"
    if score < 0.6: return "Mosaic_05"
    if score < 0.7: return "Mosaic_06"
    if score < 0.8: return "Mosaic_07"
    if score < 0.9: return "Mosaic_08"
    return "Mosaic_09" # 0.9以上は完全なグリッド

def main():
    if "server_connection" not in json_input:
        sys.stderr.write("ERROR: No server connection info\n")
        return

    client = StashInterface(json_input["server_connection"])
    log.info("Mosaic Detector: Ultra-Fine Gradient Mode (01-09) starting...")

    # 管理対象タグ：これらが付いている画像は「判定済み」とみなす
    managed_tag_names = [
        "NoMosaic", "Mosaic_01", "Mosaic_02", "Mosaic_03", "Mosaic_04", 
        "Mosaic_05", "Mosaic_06", "Mosaic_07", "Mosaic_08", "Mosaic_09"
    ]
    tag_map = {name: str(client.find_tag(name, create=True)["id"]) for name in managed_tag_names}
    managed_ids = list(tag_map.values())

    # --- 画像取得 ---
    query = "query { allImages { id files { path } tags { id } } }"
    res = client.call_GQL(query)
    all_images = res.get('allImages', [])

    # 設定で再チェックモードがOFFなら、未判定分のみ処理
    try:
        full_config = client.get_configuration()
        re_check = str(full_config.get("plugins", {}).get("mosaic-detector", {}).get("ReCheckMode", "false")).lower() == "true"
    except:
        re_check = False

    targets = []
    for i in all_images:
        current_tids = [str(t["id"]) for t in i.get("tags", [])]
        if re_check or not any(tid in managed_ids for tid in current_tids):
            targets.append(i)

    total = len(targets)
    log.info(f"Target count: {total} images. Starting fine-grained analysis...")

    # --- 解析ループ ---
    for count, item in enumerate(targets, 1):
        if not item.get("files"): continue
        
        img_id = item["id"]
        path = item["files"][0]["path"]
        current_tids = [str(t["id"]) for t in item.get("tags", [])]

        score, step = is_mosaic(path)
        tag_name = get_mosaic_tag_name(score)
        new_tag_id = tag_map[tag_name]

        # 既存の管理タグを除去して、新しい1つを付与
        final_tags = [tid for tid in current_tids if tid not in managed_ids] + [new_tag_id]

        if set(current_tids) != set(final_tags):
            mutation = """
            mutation Update($id: ID!, $tags: [ID!]) {
              imageUpdate(input: { id: $id, tag_ids: $tags }) { id }
            }
            """
            client.call_GQL(mutation, {"id": img_id, "tags": final_tags})

        log.info(f"[{count}/{total}] score:{score:.3f} -> {tag_name} | {os.path.basename(path)}")

    log.info("--- 10-Stage Analysis Completed ---")

if __name__ == "__main__":
    main()
