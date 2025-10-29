import gradio as gr
from app import App
from caption import build_system_text, normalize_caption
import glob
import json
from whisper import ANIME_WHISPER, WHISPER_LARGE_V3_TURBO

DEFAULT_LLM = "./models/Anime-Llasa-3B-Captions.Q8_0.gguf"
DEFAULT_LLM_NAME = DEFAULT_LLM[9:-5]
app = App()
app.load_llm(DEFAULT_LLM)

def load_preset_from_name(preset_name:str) -> dict:
    return json.load(open(preset_name, "r", encoding="utf-8"))

with gr.Blocks() as server:
    with gr.Row():
        with gr.Column(scale=2):
            t2s_text = gr.Textbox(label="Text to Speech", value="サンプル文章です。日本語で出力されているか確認してください。",lines=2, scale=3, max_lines=3)

            with gr.Accordion("Special Tags Cheat Sheet", open=False):
                gr.Markdown("""
　                              
[読み上げテキストについて、全角括弧を使った以下のような制御タグを利用できます。 ただしタグは自動アノテーションなので出現していないものもあると思われ、効果はものによると思われます。](https://huggingface.co/spaces/OmniAICreator/Anime-Llasa-3B-Captions-Demo)
### 1. 声の変化（スタイル・感情・意図）
- 感情/トーン：（優しく） （囁き） （自信なさげに） （からかうように） （挑発するように） （独り言のように）
- 感情の推移：（徐々に怒りを込めて） （だんだん悲しげに） （喜びを爆発させて）
- 声の状態：（声が震えて） （眠そうに） （酔っ払って） （声を枯らして）
### 2. 非言語的な発声
- 感情的な発声：（うめき声） （吐息） （息切れ） （嗚咽） （くすくす笑い） （小さな悲鳴）
- 呼吸：（息をのむ） （深い溜息） （荒い息遣い）
- 口の音：（舌打ち） （リップノイズ） （唾を飲み込む音）
### 3. アクション
- 話者の動作：（笑いながら） （泣きながら） （咳き込みながら） （勢いに任せて攻撃）
- 受ける動作：（持ち上げられて） （首を絞められて） （腹を押しつぶされて）
### 4. 音響・効果音
- 接触音：（キス音） （耳舐め） （打撃音） （衣擦れの音）
- NSFW関連音：（チュパ音） （フェラ音） （ピストン音） （射精音） （粘着質な水音）
- 環境音：（ドアの開閉音） （足音） （雨音）
- 音響効果：（電話越しに） （スピーカー越しに） （エコー）
### 5. 発話のリズム・間
- ペース：（早口で） （ゆっくりと強調して） （一気にまくしたてて）
- 間：（少し間を置いて） （一呼吸おいて） （沈黙）
### 6. 距離感・位置関係
- 位置：（遠くから） （耳元で） （背後から） （ドア越しに）
### 例
- （囁き）ふふ…今日はよく頑張ったね。（キス音）
- （徐々に怒りを込めて）もう一度言う。
                """)

            output_folder = gr.Textbox(label="Output folder name", value="",lines=1, scale=3, max_lines=1)
            max_tokens_slider =  gr.Slider(minimum=128, maximum=4096, value=512, step=64, precision=0, label="Max Tokens (trained with 2048)")

            with gr.Accordion("System Metadata"):
                presets = [f'{f[18:-5]}: {load_preset_from_name(f)["caption"]}' for f in glob.glob("./caption_presets/*.json")]
                caption_presets = gr.Dropdown(presets, label="presets", interactive=True)                
                caption_input = gr.Textbox(label="caption (REQUIRED)", placeholder="音声を説明する短いキャプション", lines=2)
                with gr.Row():
                    with gr.Column():
                        emotion_input = gr.Dropdown(["angry", "sad", "disdainful", "excited", "surprised", "satisfied", "unhappy", "anxious", "hysterical", "delighted", "scared", "worried", "indifferent", "upset", "impatient", "nervous", "guilty", "scornful", "frustrated", "depressed", "panicked", "furious", "empathetic", "embarrassed", "reluctant", "disgusted", "keen", "moved", "proud", "relaxed", "grateful", "confident", "interested", "curious", "confused", "joyful", "disapproving", "negative", "denying", "astonished", "serious", "sarcastic", "conciliative", "comforting", "sincere", "sneering", "hesitating", "yielding", "painful", "awkward", "amused", "loving", "dating", "longing", "aroused", "seductive", "ecstatic", "shy"], label="emotion", interactive=True)
                        profile_input = gr.Textbox(label="profile（お姉さん的な女性声、若い男性声、大人の女性声など）", placeholder="話者プロファイル（お姉さん的な女性声、若い男性声、大人の女性声など）", lines=1)
                        mood_input = gr.Textbox(label="mood（シリアス、快楽、恥ずかしさ、落ち着き、官能的など）", placeholder="ムード（シリアス、快楽、恥ずかしさ、落ち着き、官能的など）", lines=1)
                        speed_input = gr.Textbox(label="speed（ゆっくり、速い、一定、(1.2x) など）", placeholder="話速（ゆっくり、速い、一定、(1.2x) など）", lines=1)
                    with gr.Column():
                        prosody_input = gr.Textbox(label="prosody（メリハリがある、ため息混じり、震え声、平坦、語尾が上がるなど）", placeholder="抑揚・リズム（メリハリがある、ため息混じり、震え声、平坦、語尾が上がるなど）", lines=1)
                        pitch_timbre_input = gr.Textbox(label="pitch_timbre（高め、低め、中低音、息多め、張りのある、囁き、鼻にかかった声など）", placeholder="ピッチ・声質（高め、低め、中低音、息多め、張りのある、囁き、鼻にかかった声など）", lines=1)
                        style_input = gr.Textbox(label="style（ナレーション風、会話調、朗読調、囁き、喘ぎ、嗚咽、告白など）", placeholder="スタイル（ナレーション風、会話調、朗読調、囁き、喘ぎ、嗚咽、告白など）", lines=1)
                        notes_input = gr.Textbox(label="notes（間・ブレス・笑い、効果音の有無、距離感（耳元・遠くから）など。不要なら空でOK）", placeholder="特記事項（距離感、吐息などの追加事項）", lines=2)

            with gr.Accordion("LLM settings", open=False):
                top_k_slider =  gr.Slider(minimum=0, maximum=100, value=0, step=1, precision=0, label="Top k")
                top_p_slider =  gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, precision=2, label="Top p")
                temperature_slider =  gr.Slider(minimum=0, maximum=2, value=0.8, step=0.01, precision=1, label="Temperature")
                repeat_penalty_slider = gr.Slider(minimum=0, maximum=10, value=1.1, step=0.1, precision=1, label="Repeat Penalty")
                llms = [f[9:-5] for f in glob.glob("./models/*.gguf")]
                llm_dropdown = gr.Dropdown(llms, value=DEFAULT_LLM_NAME if DEFAULT_LLM_NAME in llms else llms[0], label="LLM", interactive=True)
                with gr.Row():
                    loras = ["None"] + [f[8:] for f in glob.glob("./loras/*")]
                    lora_path = gr.Dropdown(loras, value="None", label="LoRA", interactive=True, scale=1)
                    lora_scale = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, precision=2, label="LoRA Scale", scale=2, interactive=True)

                system_prompt = gr.Textbox(label="System prompt", value="Convert the text to speech:", lines=2, max_lines=3, scale=3, interactive=True)

            '''
            Reference Audio
            '''
            with gr.Accordion("Reference Audio (Optional)", open=True):
                with gr.Row():
                    with gr.Column(scale=4):
                        transcript_text = gr.Textbox(label="Reference Text", placeholder="If you provide reference audio, you can optionally provide its transcript here.  Reference Audio を追加した場合、その文字起こし内容をここに入力する", lines=2, max_lines=2, interactive=True)
                    with gr.Column(scale=1):
                        transcript_button = gr.Button(value="Auto Transcript", variant="primary")
                        transcript_dropdown = gr.Dropdown([ANIME_WHISPER, WHISPER_LARGE_V3_TURBO] , label="Transcript Model", interactive=True)

                user_audio = gr.Audio(label="Reference Audio (Optional)", interactive=True, type="filepath", max_length=300)
            
        '''
        右側
        '''
        with gr.Column(scale=1):
            with gr.Row():
                gen_button = gr.Button(value="Generate", variant="primary", scale=4)
                is_persistent_check = gr.Checkbox(value=False, label="Generate forever", scale=1)
            reason_label = gr.Label(label="End reason")
            audio_output = gr.Audio(editable=True, type="filepath")
    
    def t2s_gen(
            t2s_text:str, 
            system_prompt:str, 
            max_tokens:int, 
            top_k:int, 
            top_p:float, 
            temperature:float, 
            repeat_penalty:float, 
            lora_path:str,
            lora_scale:float,
            output_folder_name:str, 
            caption_input:str,
            emotion_input:str,
            profile_input:str,
            mood_input:str,
            speed_input:str,
            prosody_input:str,
            pitch_timbre_input:str,
            style_input:str,
            notes_input:str,
            user_audio:str,
            transcript_text:str):
        t2s_text = normalize_caption(t2s_text.strip())
        caption_input = normalize_caption(caption_input.strip())
        system_text = build_system_text({
            'emotion': emotion_input,
            'profile': profile_input,
            'mood': mood_input,
            'speed': speed_input,
            'prosody': prosody_input,
            'pitch_timbre': pitch_timbre_input,
            'style': style_input,
            'notes': notes_input,
            'caption': caption_input
        }) if caption_input else ""
        lora_path = f"./loras/{lora_path}" if lora_path != "None" else lora_path
        return app.t2speech(t2s_text, system_prompt, system_text, max_tokens, top_k, top_p, temperature, repeat_penalty, lora_path, lora_scale, output_folder_name, user_audio, normalize_caption(transcript_text))
    gen_button.click(fn=t2s_gen, inputs=[
        t2s_text, 
        system_prompt, 
        max_tokens_slider, 
        top_k_slider, 
        top_p_slider, 
        temperature_slider, 
        repeat_penalty_slider, 
        lora_path,
        lora_scale,
        output_folder, 
        caption_input,
        emotion_input,
        profile_input,
        mood_input,
        speed_input,
        prosody_input,
        pitch_timbre_input,
        style_input,
        notes_input,
        user_audio,
        transcript_text], outputs=[audio_output, reason_label])

    def caption_presets_change(caption_preset:str):
        caption_name = caption_preset.split(':')[0]
        j = load_preset_from_name(f'./caption_presets/{caption_name}.json')
        #gr.Info("Helpful info message ℹ️", duration=5)
        return (j["caption"], j["emotion"], j["profile"], j["mood"], j["speed"], j["prosody"], j["pitch_timbre"], j["style"], j["notes"])
    caption_presets.change(fn=caption_presets_change, inputs=caption_presets, outputs=[caption_input, emotion_input, profile_input, mood_input, speed_input, prosody_input, pitch_timbre_input, style_input, notes_input])

    llm_dropdown.change(fn=lambda model_name: app.load_llm(f"./models/{model_name}.gguf"), inputs=llm_dropdown, outputs=None)
    is_persistent_check.change(fn=lambda x: app.set_persistent(x), inputs=[is_persistent_check], outputs=None)

    def transcript_click(user_audio:str, transcript_model:str):
        if not user_audio:
            return ""
        return app.audio2text(user_audio, transcript_model)
    transcript_button.click(fn=transcript_click, inputs=[user_audio, transcript_dropdown], outputs=[transcript_text])

    
server.launch(inbrowser=True)
