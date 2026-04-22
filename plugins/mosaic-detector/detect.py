import sys
import json
import os

# 1. まず標準エラーに書き出す（これなら絶対にStashのログに残る）
sys.stderr.write("DEBUG: PYTHON ALIVE - STARTING BOOTSTRAP\n")

try:
    # 2. stdinを読み込まずにまずインポートを試す
    from PythonDepManager import ensure_import
    ensure_import("stashapi")
    import stashapi.log as log
    sys.stderr.write("DEBUG: STASHAPI IMPORT SUCCESS\n")
except Exception as e:
    sys.stderr.write(f"DEBUG: IMPORT ERROR: {e}\n")

# 3. stdinの読み込みを「お試し」で行う
raw_input = ""
try:
    # タイムアウト回避のため、読み込みを試みる
    raw_input = sys.stdin.read()
    sys.stderr.write(f"DEBUG: STDIN READ SUCCESS (LEN: {len(raw_input)})\n")
except Exception as e:
    sys.stderr.write(f"DEBUG: STDIN READ FAILED: {e}\n")

def main():
    if not raw_input:
        sys.stderr.write("DEBUG: NO INPUT FROM STDIN - EXITING CLEANLY\n")
        return

    try:
        json_input = json.loads(raw_input)
        sys.stderr.write("DEBUG: JSON LOAD SUCCESS\n")
        # ここから本来の処理へ...
        # 一旦、生存確認のためここで終わる
        sys.stderr.write("DEBUG: BOOTSTRAP COMPLETE\n")
    except Exception as e:
        sys.stderr.write(f"DEBUG: JSON LOAD ERROR: {e}\n")

if __name__ == "__main__":
    main()
