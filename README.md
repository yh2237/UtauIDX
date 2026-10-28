# UtauIDX

音声の特徴量から索引を作り、参照音声に近いフレームを選んで合成するためのライブラリです。

- `hubert`: 音声エンコーダ（WavLM base+ / HuBERT / XLS-R を config.json で読み分け）
- `refmatch`: 索引の形式、k近傍によるフレーム選択、WORLDパラメータの平均と合成
- `audio`: PCM と WAV の入出力
- `atomicfile`: 書き込み途中のファイルを残さない保存

WORLD は `refmatch.World` インターフェース越しに使うため、このモジュールは純 Go で完結します。

## ライセンス

MIT License（[LICENSE](LICENSE)）。同梱・配布するモデルのライセンスは、利用するアプリケーション側で表記してください。