import gradio as gr
from utils import App
import glob

DEFAULT_LLM = "./models/Anime-Llasa-3B.Q8_0.gguf"
DEFAULT_LLM_NAME = DEFAULT_LLM[9:-5]
app = App()
app.load_llm(DEFAULT_LLM)

with gr.Blocks() as server:
    with gr.Row():
        with gr.Column():
            t2s_text = gr.Textbox(label="Text to Speech", value="サンプル文章です。日本語で出力されているか確認してください。",lines=2, scale=3, max_lines=3)
            output_folder = gr.Textbox(label="Output folder name", value="",lines=1, scale=3, max_lines=1)
            max_tokens_slider =  gr.Slider(minimum=128, maximum=4096, value=512, step=64, precision=0, label="Max Tokens (trained with 2048)")
            with gr.Accordion("LLM settings", open=False):
                top_k_slider =  gr.Slider(minimum=0, maximum=100, value=0, step=1, precision=0, label="Top k")
                top_p_slider =  gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, precision=2, label="Top p")
                temperature_slider =  gr.Slider(minimum=0, maximum=2, value=0.9, step=0.01, precision=1, label="Temperature")
                repeat_penalty_slider = gr.Slider(minimum=0, maximum=10, value=1.1, step=0.1, precision=1, label="Repeat Penalty")
                llms = [f[9:-5] for f in glob.glob("./models/*.gguf")]
                llm_dropdown = gr.Dropdown(llms, value=DEFAULT_LLM_NAME if DEFAULT_LLM_NAME in llms else llms[0], label="LLM", interactive=True)
                system_prompt = gr.Textbox(label="System prompt", value="Convert the text to speech:", lines=2, max_lines=3, scale=3, interactive=True)
            user_audio = gr.Audio(label="Upload an audio file you want to continue generating", interactive=True, type="filepath", max_length=300)
            
        with gr.Column():
            with gr.Row():
                gen_button = gr.Button(value="Generate", variant="primary", scale=4)
                is_persistent_check = gr.Checkbox(value=False, label="Generate forever", scale=1)
            audio_output = gr.Audio(editable=True, type="filepath")
    
    def t2s_gen(t2s_text:str, system_prompt:str, max_tokens:int, top_k:int, top_p:float, temperature:float, repeat_penalty:float, output_folder_name:str, user_audio:str):
        return app.t2speech(t2s_text, system_prompt, max_tokens, top_k, top_p, temperature, repeat_penalty, output_folder_name, user_audio)
    gen_button.click(fn=t2s_gen, inputs=[t2s_text, system_prompt, max_tokens_slider, top_k_slider, top_p_slider, temperature_slider, repeat_penalty_slider, output_folder, user_audio], outputs=audio_output)

    llm_dropdown.change(fn=lambda model_name: app.load_llm(f"./models/{model_name}.gguf"), inputs=llm_dropdown, outputs=None)
    is_persistent_check.change(fn=lambda x: app.set_persistent(x), inputs=[is_persistent_check], outputs=None)
server.launch(inbrowser=True)
