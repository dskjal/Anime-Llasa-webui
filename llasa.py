from llama_cpp import Llama

import torch
import time
import numpy as np

# トークンマップは https://huggingface.co/HKUSTAudio/Llasa-3B/blob/main/tokenizer.json を参照
TOKEN_OFFSET = 128264
EOT_ID = 128009
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
        path = path or self.llm_path
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

        if audio_size != cache_size:
            return False
        
        for i in range(audio_size):
            if self.audio_token_cache[i] != audio_tokens[i]:
                return False
        return True


    def t2token(self, t2s_text:str, system_prompt:str="Convert the text to speech:", system_text:str="", max_tokens:int=2048, top_k:int=0, top_p:float=0.95, temperature:float=0.7, repeat_penalty:float=1.1, audio_tokens:np.array=np.empty(0), transcript_text:str="") -> list:
        if not self.is_loaded():
            self.load()

        t2s_text = transcript_text + t2s_text   # 後方互換性のため
        
        with torch.no_grad():
            # cache check
            prompt_tokens = self.prompt_cache
            if self.t2s_text_cache == t2s_text and self.system_text_cache == system_text and self.prompt_cache and self.is_same_audio(audio_tokens):
                # use cache
                print("Use prompt cache")
            else:
                text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{system_prompt}<|TEXT_UNDERSTANDING_START|>{t2s_text}<|TEXT_UNDERSTANDING_END|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|SPEECH_GENERATION_START|>"

                prompt_tokens = self.llm.tokenize(text.encode('utf-8'), add_bos=False, special=True)

                if len(audio_tokens) > 0:
                    prompt_tokens.extend([token+TOKEN_OFFSET for token in audio_tokens.tolist()])
                self.audio_token_cache = audio_tokens

                self.t2s_text_cache = t2s_text
                self.system_text_cache = system_text
                self.audio_token_cache = audio_tokens
                self.prompt_cache = prompt_tokens

            self.llm.reset()
            self.llm.eval(prompt_tokens)

            # https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py#L1315C1-L1318C73
            # if seed != -1:
            #     self.llm.set_seed(seed)
            # else:
            #     import random
            #     self.llm.set_seed(random.Random(self.llm._seed).randint(0, 2 ** 32))

            reason = "Max token reached."
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
                    reason = "Success. (EOS token detected)"
                    print("\n[EOS token detected]")
                    break
                
                if len(generated_tokens) >= 100 and len(set(generated_tokens[-100:])) < 20:
                    reason = "Repetition detected."
                    print("\nRepetition detected.")
                    break
                
                if token >= TOKEN_OFFSET:
                    generated_tokens.append(token-TOKEN_OFFSET)

                # トークン追加
                self.llm.eval([token])
            return generated_tokens, reason