from llama_cpp import Llama 
import torch
import soundfile as sf
import librosa
import numpy as np
import os
from xcodec2.modeling_xcodec2 import XCodec2Model

# トークンマップは https://huggingface.co/HKUSTAudio/Llasa-3B/blob/main/tokenizer.json を参照
TEXT_UNDERSTANDING_START = 128258
TEXT_UNDERSTANDING_END = 128259
SPEECH_GENERATION_START = 128260
SPEECH_GENERATION_END = 128261
BEGIN_OF_TEXT = 128000
END_OF_TEXT = 128001
EOT_ID = 128009
START_HEADER_ID = 128006
END_HEADER_ID = 128007
MAX_CONTEXT_SIZE = 4096

TARGET_SR = 16000   # xcodec2 の動作周波数
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

        model_path = "HKUSTAudio/xcodec2"
        self.Codec_model = XCodec2Model.from_pretrained(model_path)
        self.Codec_model.eval()

    def load_llm(self, path:str="./models/Anime-Llasa-3B.Q4_K_M.gguf"):
        if not os.path.isfile(path=path):
            raise FileNotFoundError(path)
        
        if self.llm != None:
            self.unload_llm()

        try:
            self.llm = Llama(
                model_path=path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=MAX_CONTEXT_SIZE,
                verbose=False,
            )
            self.llm_path = path
            self.prompt_cache = []
            print(f"{path} loaded.")
            return
        except Exception as e:
            print(f"Failed to load llm: {e}")

        # try offload xcodec2
        self.Codec_model.to(cpu_device)
        torch.cuda.empty_cache()
        try:
            self.llm = Llama(
                model_path=path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=MAX_CONTEXT_SIZE,
                verbose=False,
            )
            self.llm_path = path
            self.prompt_cache = []
            print(f"{path} loaded.")
        except Exception as e:
            print(f"XCodec2 offloaded. But failed to load llm: {e}")

    
    def unload_llm(self):
        self.llm.close()
        self.llm = None
        print("unload llm")

    def set_persistent(self, is_persistent):
        self.is_persistent_generation = is_persistent
        
    def t2speech(self, t2s_text:str, system_prompt:str="Convert the text to speech:", max_tokens:int=2048, top_k:int=0, top_p:float=0.95, temperature:float=0.9, repeat_penalty:float=1.1, output_folder_name="", audio_file_name:str="") -> str:
        import time
        audio_tokens = self.audio_token_cache
        if audio_file_name:
            if audio_file_name != self.audio_file_name_cache:
                old_time = time.time()
                audio_tokens = self.speech2token(audio_file_name)
                self.audio_file_name_cache = audio_file_name
                # audio_token_cache は t2token で管理
                print(f"Audio to tokens : {time.time()-old_time:.1f} sec")
        else:
            audio_tokens = self.audio_token_cache = np.empty(0)

        while True:
            old_time = time.time()
            tokens = self.t2token(t2s_text, system_prompt, max_tokens, top_k, top_p, temperature, repeat_penalty, audio_tokens)
            print(f"Inference : {time.time()-old_time:.1f} sec")

            old_time = time.time()
            filepath =  self.token2speech(tokens, output_folder_name)
            print(f"Speech generation : {time.time()-old_time:.1f} sec\n")

            if not self.is_persistent_generation:
                break

        return filepath

    def is_same_audio(self, audio_tokens:np.array):
        if len(self.audio_token_cache) == 0 and len(audio_tokens) == 0:
            return True
        
        if len(self.audio_token_cache) == 0 or len(audio_tokens) == 0:
            return False
        
        if len(self.audio_token_cache) != len(audio_tokens):
            return False
        
        for i in range(len(audio_tokens)):
            if self.audio_token_cache[i] != audio_tokens[i]:
                return False
        return True
    
    def t2token(self, t2s_text:str, system_prompt:str="Convert the text to speech:", max_tokens:int=2048, top_k:int=0, top_p:float=0.95, temperature:float=0.9, repeat_penalty:float=1.1, audio_tokens:np.array=np.empty(0)) -> list:
        if self.llm == None and self.llm_path == None:
            raise RuntimeError("LLM is not loaded.")
        
        if self.llm == None and self.llm_path:
            self.load_llm(self.llm_path)

        with torch.no_grad():
            self.llm.reset()

            # cache check
            prompt_tokens = self.prompt_cache
            if self.t2s_text_cache == t2s_text and self.system_prompt_cache == system_prompt and self.prompt_cache and self.is_same_audio(audio_tokens):
                # use cache
                print("Use prompt cache")
            else:
                # <|begin_of_text|><|start_header_id|>user<|end_header_id|>user content<|eot_id|><|start_header_id|>assistant<|end_header_id|>assistant content<|eot|>
                prompt_tokens = []
                # prompt_tokens.append(BEGIN_OF_TEXT)

                # prompt_tokens.append(START_HEADER_ID)
                # prompt_tokens.extend(self.llm.tokenize("user".encode('utf-8')))
                # prompt_tokens.append(END_HEADER_ID)
                prompt_tokens.extend(self.llm.tokenize(system_prompt.encode('utf-8')))
                prompt_tokens.append(TEXT_UNDERSTANDING_START)
                prompt_tokens.extend(self.llm.tokenize(t2s_text.encode('utf-8')))
                prompt_tokens.append(TEXT_UNDERSTANDING_END)
                # prompt_tokens.append(EOT_ID)

                # prompt_tokens.append(START_HEADER_ID)
                # prompt_tokens.extend(self.llm.tokenize("assistant".encode('utf-8')))
                # prompt_tokens.append(END_HEADER_ID)
                prompt_tokens.append(SPEECH_GENERATION_START)

                if len(audio_tokens) > 0:
                    prompt_tokens.extend(audio_tokens.tolist())

                self.t2s_text_cache = t2s_text
                self.system_prompt_cache = system_prompt
                self.audio_token_cache = audio_tokens
                self.prompt_cache = prompt_tokens

            self.llm.eval(prompt_tokens)

            generated_tokens = audio_tokens.tolist()
            for i in range(max_tokens):
                token = self.llm.sample(
                    top_k=top_k,
                    top_p=top_p,
                    temp=temperature,
                    repeat_penalty=repeat_penalty,
                )

                TOKEN_OFFSET = 128264
                if token >= TOKEN_OFFSET:
                    generated_tokens.append(token-TOKEN_OFFSET)

                # 停止条件
                if token == SPEECH_GENERATION_END:
                    print("\n[EOS token detected]")
                    break
                elif token == END_OF_TEXT:
                    print("\n[end_of_text token detected]")
                    break

                # トークン追加
                self.llm.eval([token])
            return generated_tokens

    def load_xcode2(self) -> None:
        if self.Codec_model.device != cuda_device:
            # VRAM check
            free, total = torch.cuda.mem_get_info(cuda_device)
            free_vram_GiB = free/1024**3
            if free_vram_GiB <= 3.5:
                self.unload_llm()

            self.Codec_model.to(cuda_device)
    
    def speech2token(self, audio_file:str) -> np.array:
        if not os.path.isfile(audio_file):
            raise FileNotFoundError(audio_file)
        
        self.load_xcode2()
        wav, _ = librosa.load(audio_file, sr=TARGET_SR)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0) # Shape: (1, T)
        return self.Codec_model.encode_code(input_waveform=wav_tensor).cpu()[0, 0, :].numpy()
        
    def token2speech(self, tokens:list, output_folder_name:str="") -> str:
        self.load_xcode2()

        speech_tokens = torch.tensor(tokens, device=cuda_device).unsqueeze(0).unsqueeze(0)
        gen_wav = self.Codec_model.decode_code(speech_tokens)

        if output_folder_name:
            if not os.path.isdir(f"./outputs/{output_folder_name}"):
                os.mkdir(f"./outputs/{output_folder_name}")
            output_folder_name = output_folder_name + "/"

        import datetime
        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_file = f"./outputs/{output_folder_name}{filename}.wav"
        sf.write(output_file, gen_wav[0, 0, :].cpu().numpy(), TARGET_SR)
        # with open(f"./outputs/{filename}.txt", "w", encoding="utf-8") as f:
        #     f.write(','.join([str(i) for i in tokens]))
        return output_file