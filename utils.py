from llama_cpp import Llama
import torch
import soundfile as sf
import librosa
import numpy as np
import os
import time
from xcodec2.modeling_xcodec2 import XCodec2Model

# トークンマップは https://huggingface.co/HKUSTAudio/Llasa-3B/blob/main/tokenizer.json を参照
TOKEN_OFFSET = 128264
EOT_ID = 128009
TEXT_UNDERSTANDING_START = 128258
TEXT_UNDERSTANDING_END = 128259
SPEECH_GENERATION_START = 128260
SPEECH_GENERATION_END = 128261
MAX_CONTEXT_SIZE = 4096


USE_FP16_XCODEC2 = True
DTYPE = torch.float32
if USE_FP16_XCODEC2:
    DTYPE = torch.float16

USE_44KHZ = True
XCODEC2_MODEL = "HKUSTAudio/xcodec2"
ENCODER_SR = 16000   # xcodec2 の動作周波数
DECODER_SR = ENCODER_SR
if USE_44KHZ:
    XCODEC2_MODEL = "NandemoGHS/Anime-XCodec2-44.1kHz"
    DECODER_SR = 44100

cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else cpu_device

class App:
    def __init__(self):
        self.n_gpu_layers = -1 if torch.cuda.is_available() else 0 
        self.llm = None
        self.llm_path = None
        self.is_persistent_generation = False
        self.t2s_text_cache = ""
        self.system_prompt_cache = ""
        self.audio_file_name_cache = ""
        self.audio_token_cache = np.empty(0)
        self.prompt_cache = []

        old_time = time.time()
        self.Codec_model = XCodec2Model.from_pretrained(XCODEC2_MODEL)
        self.Codec_model.eval()

        if USE_FP16_XCODEC2:
            # fp16 に変換
            self.Codec_model = self.Codec_model.half()
            # LayerNorm モジュールを特定し、FP32 に戻す
            def cast_ln_to_fp32(m):
                # PyTorchの LayerNorm のインスタンスであれば
                if isinstance(m, torch.nn.LayerNorm):
                    m.float() # LayerNorm の重みを FP32 にキャスト
                    
            # モデル全体に適用
            self.Codec_model.apply(cast_ln_to_fp32)

        print(f"XCodec2 load time : {time.time()-old_time:.1f} sec")


    def do_load_llm(self, path:str) -> None:
        old_time = time.time()
        self.llm = Llama(
            model_path=path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=MAX_CONTEXT_SIZE,
            flash_attn=True,
            # type_k=8,   # GGML_TYPE_Q8_0
            # type_v=8,   # GGML_TYPE_Q8_0
            verbose=False,
        )
        self.llm_path = path
        self.prompt_cache = []
        print(f"{path} loaded. Load time : {time.time()-old_time:.1f} sec")

    def load_llm(self, path:str):
        if not os.path.isfile(path=path):
            raise FileNotFoundError(path)
        
        if self.llm != None:
            self.unload_llm()

        try:
            self.do_load_llm(path)
            return
        except Exception as e:
            print(f"Failed to load llm: {e}")

        # try offload xcodec2
        self.Codec_model.to(cpu_device)
        torch.cuda.empty_cache()
        try:
            self.do_load_llm(path)
        except Exception as e:
            print(f"XCodec2 offloaded. But failed to load llm: {e}")

    
    def unload_llm(self):
        self.llm.close()
        self.llm = None
        print("unload llm")

    def set_persistent(self, is_persistent):
        self.is_persistent_generation = is_persistent
        
    def t2speech(self, t2s_text:str, system_prompt:str="Convert the text to speech:", system_text:str="", max_tokens:int=2048, top_k:int=0, top_p:float=0.95, temperature:float=0.7, repeat_penalty:float=1.1, output_folder_name="", audio_file_name:str="") -> str:
        audio_tokens = self.audio_token_cache
        if audio_file_name:
            if audio_file_name != self.audio_file_name_cache:
                old_time = time.time()
                audio_tokens = self.speech2token(audio_file_name)
                self.audio_file_name_cache = audio_file_name
                # audio_token_cache は t2token で管理
                print(f"Audio to tokens : {time.time()-old_time:.1f} sec")
        else:
            audio_tokens = np.empty(0)

        while True:
            old_time = time.time()
            tokens = self.t2token(t2s_text, system_prompt, system_text, max_tokens, top_k, top_p, temperature, repeat_penalty, audio_tokens)
            print(f"Inference : {time.time()-old_time:.1f} sec")

            old_time = time.time()
            filepath =  self.token2speech(tokens, output_folder_name)
            print(f"Speech generation : {time.time()-old_time:.1f} sec\n")

            if not self.is_persistent_generation:
                break

        return filepath

    def is_same_audio(self, audio_tokens:np.array):
        cache_size = len(self.audio_token_cache)
        audio_size = len(audio_tokens)

        if audio_size == cache_size:
            for i in range(audio_size):
                if self.audio_token_cache[i] != audio_tokens[i]:
                    return False
            return True
        return False
    
    def t2token(self, t2s_text:str, system_prompt:str="Convert the text to speech:", system_text:str="", max_tokens:int=2048, top_k:int=0, top_p:float=0.95, temperature:float=0.7, repeat_penalty:float=1.1, audio_tokens:np.array=np.empty(0)) -> list:
        if self.llm == None and self.llm_path == None:
            raise RuntimeError("LLM is not loaded.")
        
        if self.llm == None and self.llm_path:
            self.load_llm(self.llm_path)

        with torch.no_grad():
            self.llm.reset()

            # cache check
            prompt_tokens = self.prompt_cache
            if self.t2s_text_cache == t2s_text and self.system_text_cache == system_text and self.prompt_cache and self.is_same_audio(audio_tokens):
                # use cache
                print("Use prompt cache")
            else:
                prompt_tokens = []
                prompt_tokens.extend(self.llm.tokenize(system_text.encode('utf-8')))
                prompt_tokens.extend(self.llm.tokenize(system_prompt.encode('utf-8')))
                prompt_tokens.append(TEXT_UNDERSTANDING_START)
                prompt_tokens.extend(self.llm.tokenize(t2s_text.encode('utf-8')))
                prompt_tokens.append(TEXT_UNDERSTANDING_END)
                prompt_tokens.append(SPEECH_GENERATION_START)

                if len(audio_tokens) > 0:
                    prompt_tokens.extend([token+TOKEN_OFFSET for token in audio_tokens.tolist()])
                self.audio_token_cache = audio_tokens

                self.t2s_text_cache = t2s_text
                self.system_text_cache = system_text
                self.audio_token_cache = audio_tokens
                self.prompt_cache = prompt_tokens

            self.llm.eval(prompt_tokens)

            generated_tokens = []#audio_tokens.tolist()
            for _ in range(max_tokens):
                token = self.llm.sample(
                    top_k=top_k,
                    top_p=top_p,
                    temp=temperature,
                    repeat_penalty=repeat_penalty
                )
                # 停止条件
                if token == SPEECH_GENERATION_END:
                    print("\n[EOS token detected]")
                    break

                if token >= TOKEN_OFFSET:
                    generated_tokens.append(token-TOKEN_OFFSET)

                # トークン追加
                self.llm.eval([token])
            return generated_tokens

    def load_xcode2(self) -> None:
        if self.Codec_model.device != cuda_device:
            # VRAM check
            free, total = torch.cuda.mem_get_info(cuda_device)
            free_vram_GiB = free/1024**3
            if free_vram_GiB <= 2   :    # fp32 の場合は 3.5
                self.unload_llm()

            self.Codec_model.to(cuda_device)
    
    def speech2token(self, audio_file:str) -> np.array:
        if not os.path.isfile(audio_file):
            raise FileNotFoundError(audio_file)
        
        self.load_xcode2()
        wav, _ = librosa.load(audio_file, sr=ENCODER_SR)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0) # Shape: (1, T)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                return self.Codec_model.encode_code(input_waveform=wav_tensor).cpu()[0, 0, :].numpy()
        
    def token2speech(self, tokens:list, output_folder_name:str="") -> str:
        self.load_xcode2()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                speech_tokens = torch.tensor(tokens, device=cuda_device).unsqueeze(0).unsqueeze(0)
                gen_wav = self.Codec_model.decode_code(speech_tokens)

        if output_folder_name:
            if not os.path.isdir(f"./outputs/{output_folder_name}"):
                os.mkdir(f"./outputs/{output_folder_name}")
            output_folder_name = output_folder_name + "/"

        import datetime
        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_file = f"./outputs/{output_folder_name}{filename}.wav"
        sf.write(output_file, gen_wav[0, 0, :].cpu().numpy(), DECODER_SR)
        # with open(f"./outputs/{filename}.txt", "w", encoding="utf-8") as f:
        #     f.write(','.join([str(i) for i in tokens]))
        return output_file