# Anime-Llasa-webui
VRAM 8 GB で Anime Llasa を動作させる webui

![](https://github.com/dskjal/Anime-Llasa-webui/blob/main/images/ui.jpg)

# インストール

## 必要なもの
- git
- python 3.11
- [Microsoft Visual C++ 2015-2022 Redistributable](https://learn.microsoft.com/ja-jp/cpp/windows/latest-supported-vc-redist?view=msvc-170)
- [Anime-Llasa-3B.Q4_K_M.gguf](https://huggingface.co/mradermacher/Anime-Llasa-3B-GGUF/blob/main/Anime-Llasa-3B.Q4_K_M.gguf)

## コマンド
> git clone https://github.com/dskjal/Anime-Llasa-webui  
> cd Anime-Llasa-webui  
> python -m venv venv  
> ./venv/Scripts/activate  
> pip install -r requirements.txt

Anime-Llasa-webui/models に [Anime-Llasa-3B.Q4_K_M.gguf](https://huggingface.co/mradermacher/Anime-Llasa-3B-GGUF/blob/main/Anime-Llasa-3B.Q4_K_M.gguf) を配置。

# 起動
Anime-Llasa-webui ディレクトリで
> ./venv/Scripts/activate  
> python ./run.py

**初回起動時は HKUSTAudio/xcodec2 のダウンロードがあるので遅い。**

# 使い方
**音声ファイルの続きを生成する機能は、実装してありますが正しく機能しません**。

**よく生成に失敗します。何度か生成してみてください**。

Generate forever にチェックを入れて、Generate を押すと無限生成します。停止するには、Generate forever のチェックを外して、現在生成中の音声が生成されるまで待ってください。

System prompt を変更した学習はされていないので、System prompt を変更しても意味はありません。

safetensor モデルの読み込みは対応していません。

### 日本語の読み上げに失敗する
- ひらがな・カタカナにする
- 読点（、）を入れる
- 別の漢字に変換する（成功、置換など）

### 音声ファイルの続きを生成する機能

音声ファイルの音声が「こんにちは」の場合、Text to Speech に「こんにちは『続きの文章』」を入力する。**音声ファイルの続きを生成する機能は、実装してありますが正しく機能しません**。

# トラブルシューティング
## xcodec2 の動作検証
tests フォルダで powershell を開き以下のコマンドを実行。reconstructed.wav が作成され、正常に再生されることを確認する。

> ..\venv\Scripts\activate  
> python .\test_xcodec2.py


## llm の動作検証
tests フォルダで powershell を開き、Anime-Llasa-webui/models に Anime-Llasa-3B.Q4_K_M.gguf があることを確認する。以下のコマンドを実行し、0～65536 の数値が大量に出力されると成功。

> ..\venv\Scripts\activate  
> python .\test_llm.py

# 動作について

[HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2) は音声ファイルを 0～65535 の整数の配列にエンコード、整数の配列から音声ファイルへデコードするモデル。

[HKUSTAudio/Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B) は Llama 3.2 を、テキストから XCodec2 がデコード可能な数値の配列を出力するようにファインチューンしたもの。

[NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B) は英語と中国語とにしか対応していない Llasa-3B を日本語に対応するように追加学習したもの。

[mradermacher/Anime-Llasa-3B-GGUF](https://huggingface.co/mradermacher/Anime-Llasa-3B-GGUF) は Anime-Llasa-3B の量子化モデル。

生成にかかる時間の 99% が Llasa のトークン生成なので、Llama 3.2 が高速に動作するハードウェアを使うか、Llama 3.2 に適用可能な最適化を行うと動作が高速になる。

# 関連リポジトリ
- [HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2)
- [HKUSTAudio/Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B)
- [NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B)
- [mradermacher/Anime-Llasa-3B-GGUF](https://huggingface.co/mradermacher/Anime-Llasa-3B-GGUF)
