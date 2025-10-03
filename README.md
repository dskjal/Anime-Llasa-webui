# Anime-Llasa-webui
VRAM 8 GB で Anime Llasa を動作させる webui

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

# トラブルシューティング
## xcodec2 の動作検証
tests フォルダで powershell を開き以下のコマンドを実行。reconstructed.wav が作成され、正常に再生されることを確認する。

> ..\venv\Scripts\activate  
> python .\test_xcodec2.py


## llm の動作検証
tests フォルダで powershell を開き、Anime-Llasa-webui/models に Anime-Llasa-3B.Q4_K_M.gguf があることを確認する。以下のコマンドを実行し、0～65536 の数値が大量に出力されると成功。

> ..\venv\Scripts\activate  
> python .\test_llm.py

# 関連リポジトリ
- [HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2)
- [HKUSTAudio/Llasa-3B](https://huggingface.co/HKUSTAudio/Llasa-3B)
- [NandemoGHS/Anime-Llasa-3B](https://huggingface.co/NandemoGHS/Anime-Llasa-3B)
- [mradermacher/Anime-Llasa-3B-GGUF](https://huggingface.co/mradermacher/Anime-Llasa-3B-GGUF)
