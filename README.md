# UtauIDX

音声の特徴量から索引を作り、参照音声に近いフレームを選んで合成するためのライブラリです。

- `encoder`: 音声特徴量エンコーダ（`WavLM`・`Wav2Vec2-Espeak`・`Whisper`、および外部推論ストア）
- `speechpool`: `oto.ini` の走査、言語非依存の疑似発話（Oto-Babble）生成、発話プール管理
- `index`: マルチチャンネル索引形式 `UTSP v1` と分離部分キャッシュ
- `matcher`: `gap` / `xling` 自動制御とマルチチャンネル類似度照合
- `vocoder`: `WorldVocoder` による音響パラメータ復号と WORLD 合成
- `audio`: PCM と WAV の入出力（`audiodsp/wav` の再公開とトリミング）
- `atomicfile`: 書き込み途中のファイルを残さない保存

WORLD はインターフェース（`index.World`）越しに使うため、このモジュールは純 Go で完結します。

## ライセンス

MIT License（[LICENSE](LICENSE)）。