# Anime-Llasa-webui
VRAM 8 GB で Anime Llasa 3B Captions を動作させる webui。Anime-XCodec2-44.1kHz を fp16 で動作させているので、VRAM 8 GB で Anime Llasa Captions Q8_0 が使用できる。

![](https://github.com/dskjal/Anime-Llasa-webui/blob/main/images/ui.png)

# インストール

## 必要なもの
- git
- python 3.11
- [Microsoft Visual C++ 2015-2022 Redistributable](https://learn.microsoft.com/ja-jp/cpp/windows/latest-supported-vc-redist?view=msvc-170)
- [Anime-Llasa-3B-Captions.Q8_0.gguf](https://huggingface.co/dskjal/Anime-Llasa-3B-Captions-GGUF/blob/main/Anime-Llasa-3B-Captions.Q8_0.gguf)

## コマンド
> git clone https://github.com/dskjal/Anime-Llasa-webui  
> cd Anime-Llasa-webui  
> python -m venv venv  
> ./venv/Scripts/activate  
> pip install -r requirements.txt  


Anime-Llasa-webui/models に [Anime-Llasa-3B-Captions.Q8_0.gguf](https://huggingface.co/dskjal/Anime-Llasa-3B-Captions-GGUF/blob/main/Anime-Llasa-3B-Captions.Q8_0.gguf) を配置。

# 起動
Anime-Llasa-webui ディレクトリで
> ./venv/Scripts/activate  
> python ./run.py

**初回起動時は XCodec2 のダウンロードがあるので遅い。**

# 使い方

**よく生成に失敗します。何度か生成してみてください**。

### 生成方法

1. Text to Speech に読み上げたい文章を入力する
2. System Metadata の Presets から適当なプリセットを適用する
3. 右の Generate ボタンを押す
4. outputs フォルダに音声が生成される

### 音声ファイルの続きを生成する機能

1. Reference Audio にオーディオファイルをアップロードする
2. Auto Transcript で自動書き起こしをするか、手動で音声テキストを入力する。初回実行時はモデルのダウンロードが発生するので遅い
3. Text to Speech に続きの文章を入力する

音声ファイルを使う場合は古いモデルの [Anime-Llasa-3B.Q8_0.gguf](https://huggingface.co/mradermacher/Anime-Llasa-3B-GGUF/blob/main/Anime-Llasa-3B.Q8_0.gguf) を使う方がいいかもしれない。音声ファイルを使う場合はキャプション（System Metadata）が無視されることが多く、Caption モデルは生成の成功率も低い。

### キャプション（System Metadata）なしでの生成

System Metadata の Presets で @clear: を選択して、Caption (REQUIRED) を空にして生成する。

### 注意事項

Generate forever にチェックを入れて、Generate を押すと無限生成します。停止するには、Generate forever のチェックを外して、現在生成中の音声が生成されるまで待ってください。

キャプション（System Metadata）とアップロードしたオーディオファイルとが矛盾している場合、オーディオファイルの内容が優先されることが多いです。例えばキャプションが男性の声なのに、アップロードしたオーディオファイルが女性の声の場合、続きは女性の声になります。

safetensor モデルの読み込みは対応していません。

### 日本語の読み上げに失敗する
- ひらがな・カタカナにする
- 読点（、）を入れる
- 別の漢字に変換する（成功、置換など）

# トラブルシューティング
## 音声ファイルを入力すると生成時にエラーになる
以下のエラーは XCodec2 がエンコードに失敗したときに出る。環境音・吐息・笑い声のような音声化しづらい音がファイルの前後に入っていると失敗しやすい。

<pre>
einx.expr.stage3.SolveValueException: Failed to solve values of expressions. Axis 'n' has value 0 <= 0
Input:
    'q [c] d = 1 65536 8'
    'b n q = 1 0 1'
    'q b n d = None'
    '1 = 1'
</pre>

## XCodec2 の動作検証
tests フォルダで powershell を開き以下のコマンドを実行。reconstructed.wav が作成され、正常に再生されることを確認する。

> ..\venv\Scripts\activate  
> python .\test_xcodec2.py


## LLM の動作検証
tests フォルダで powershell を開き、Anime-Llasa-webui/models に Anime-Llasa-3B-Captions.Q8_0.gguf があることを確認する。以下のコマンドを実行し、0～65535 の数値が大量に出力されると成功。

> ..\venv\Scripts\activate  
> python .\test_llm.py

# 動作について

生成にかかる時間の 99% が Llasa のトークン生成なので、Llama 3.2 が高速に動作するハードウェアを使うか、Llama 3.2 に適用可能な最適化を行うと動作が高速になる。

### XCodec2

[HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2) は音声ファイルを 0～65535 の整数の配列にエンコード、整数の配列から音声ファイルへデコードするモデル。

[NandemoGHS/Anime-XCodec2-44.1kHz](https://huggingface.co/NandemoGHS/Anime-XCodec2-44.1kHz) XCodec2 の出力層を 44.1kHz に対応するように、ファインチューンしたもの。

### (Anime) Llasa 3B

[HKUSTAudio/Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B) は Llama 3.2 を、テキストから XCodec2 がデコード可能な数値の配列を出力するようにファインチューンしたもの。

[NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B) は英語と中国語とにしか対応していない Llasa-3B を日本語に対応するように追加学習したもの。

[mradermacher/Anime-Llasa-3B-GGUF](https://huggingface.co/mradermacher/Anime-Llasa-3B-GGUF) は Anime-Llasa-3B の量子化モデル。

[Anime-Llasa-3B.Q8_0.gguf](https://huggingface.co/mradermacher/Anime-Llasa-3B-GGUF/blob/main/Anime-Llasa-3B.Q8_0.gguf) 

### Anime Llasa 3B Captions

[NandemoGHS/Anime-Llasa-3B-Captions](https://huggingface.co/NandemoGHS/Anime-Llasa-3B-Captions) キャプションを使って Anime-Llasa-3B をさらにファインチューンしたもの。

[dskjal/Anime-Llasa-3B-Captions-GGUF](https://huggingface.co/dskjal/Anime-Llasa-3B-Captions-GGUF/blob/main/Anime-Llasa-3B-Captions.Q8_0.gguf) は Anime-Llasa-3B-Captions の量子化モデル。



# LoRA

LoRA の作成方法は [zhenye234/LLaSA_training](https://github.com/zhenye234/LLaSA_training/tree/main) を参照。

# SageAttention

[F.scaled_dot_product_attention = sageattn のような単純なパッチでは、精度が大きく低下すると報告されている](https://github.com/thu-ml/SageAttention/issues/55)。Llama 3.2 は Grouped-Query Attention を採用しており、SageAttention が GQA に非対応のため。

