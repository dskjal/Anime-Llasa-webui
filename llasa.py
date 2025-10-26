from llama_cpp import Llama

import torch
import time
import numpy as np
from utils import cpu_device, cuda_device, has_available_vram_gb

# トークンマップは https://huggingface.co/HKUSTAudio/Llasa-3B/blob/main/tokenizer.json を参照
TOKEN_OFFSET = 128264
EOT_ID = 128009
TEXT_UNDERSTANDING_START = 128258
TEXT_UNDERSTANDING_END = 128259
SPEECH_GENERATION_START = 128260
SPEECH_GENERATION_END = 128261
MAX_CONTEXT_SIZE = 4096

class Llasa():
    def __init__(self) -> None:
        self.n_gpu_layers = -1 if torch.cuda.is_available() else 0 
        self.llm = None
        self.llm_path = None

        # cache
        self.t2s_text_cache = ""
        self.system_prompt_cache = ""
        self.audio_file_name_cache = ""
        self.audio_token_cache = np.empty(0)
        self.prompt_cache = []

    def is_loaded(self) -> bool:
        return self.llm != None
    
    def load(self, path:str="") -> None:
        self.unload()
        self.prompt_cache = []
        if not path:
            path = self.llm_path
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
        print(f"{path} loaded. Load time : {time.time()-old_time:.1f} sec")

    def unload(self) -> None:
        if self.llm:
            self.llm.close()
            self.llm = None
            print("unload llm")

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
        if not self.is_loaded():
            self.load()

        with torch.no_grad():
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

            self.llm.reset()
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