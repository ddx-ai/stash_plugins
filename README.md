## Stash用プラグイン
Stash用プラグインのリポジトリです。
https://ddx-ai.github.io/stash_plugins/main/index.yml

このテンプレートから作成。
https://github.com/stashapp/plugins-repo-template

### mosaic-detector
モザイク画像を検出するプラグインです。
モザイクレベルに応じてNoMosaic、Mosaic_01~Mosaic_09のタグをつけます。
事前に以下のモジュールをインストールしてください。

pip install stashapi opencv-python requests numpy

#### オプション
##### 再チェックモード
- NoMosaic、Mosaic_01等のついた画像も再チェックします。
##### 角度許容誤差 (0-45)
- 大きいほど傾きのあるモザイクを対象にします。1を推奨
##### 判定開始しきい値 (0.0-1.0)
- 0.3くらいを推奨 格子模様など誤検出する場合は大きい値を設定してください
##### 対象タグ限定
- このタグが含まれる画像のみを対象にします。


