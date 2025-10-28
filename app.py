import torch
import numpy as np
import os
import time
from utils import has_available_vram_gb
from xcodec2wrapper import XCodec2Wrapper
from llasa import Llasa
from whisper import Whisper


class App:
    def __init__(self):
        self.is_persistent_generation = False

        self.xcodec2 = XCodec2Wrapper(use_fp16=True, use_44khz=True)
        self.llm = Llasa()
        self.whisper = Whisper()

        # cache
        self.audio_token_cache = np.empty(0)
        self.audio_file_name_cache = ""

    def load_llm(self, path:str):
        if not os.path.isfile(path=path):
            raise FileNotFoundError(path)
        
        try:
            self.llm.load(path)
            return
        except Exception as e:
            print(f"Failed to load llm: {e}")

        # offload xcodec2
        self.xcodec2.move_to_ram()
        torch.cuda.empty_cache()
        try:
            self.llm.load(path)
        except Exception as e:
            print(f"XCodec2 offloaded. But failed to load llm: {e}")

    def set_persistent(self, is_persistent):
        self.is_persistent_generation = is_persistent
        
    def t2speech(self, t2s_text:str, system_prompt:str="Convert the text to speech:", system_text:str="", max_tokens:int=2048, top_k:int=0, top_p:float=0.95, temperature:float=0.7, repeat_penalty:float=1.1, output_folder_name="", audio_file_name:str="", transcript_text:str="") -> str:

        # audio2token
        audio_tokens = self.audio_token_cache
        if audio_file_name:
            if audio_file_name != self.audio_file_name_cache:
                old_time = time.time()
                audio_tokens = self.xcodec2.speech2token(audio_file_name)
                self.audio_file_name_cache = audio_file_name
                self.audio_token_cache = audio_tokens
                print(f"Audio to tokens : {time.time()-old_time:.1f} sec")
        else:
            self.audio_file_name_cache = ""
            self.audio_token_cache = audio_tokens = np.empty(0)

        # generation
        while True:
            old_time = time.time()

            tokens, reason = self.llm.t2token(t2s_text, system_prompt, system_text, max_tokens, top_k, top_p, temperature, repeat_penalty, audio_tokens, transcript_text)
            print(f"Inference : {time.time()-old_time:.1f} sec")

            if not has_available_vram_gb(self.xcodec2.get_required_vram_size_gb()):
                self.llm.unload()

            if output_folder_name:
                if not os.path.isdir(f"./outputs/{output_folder_name}"):
                    os.mkdir(f"./outputs/{output_folder_name}")
                output_folder_name = output_folder_name + "/"

            import datetime
            filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            audio_file_path = f"./outputs/{output_folder_name}{filename}.wav"

            # write token
            # with open(f"./outputs/{output_folder_name}{filename}.txt", "w", encoding='utf-8') as f:
            #     for t in tokens:
            #         print(t, sep=',', file=f)

            old_time = time.time()
            self.xcodec2.token2speech(tokens, audio_file_path)
            print(f"Speech generation : {time.time()-old_time:.1f} sec\n")

            if not self.is_persistent_generation:
                break

        return audio_file_path, reason

    def audio2text(self, audio_file_name:str, transcript_model:str) -> str:
        old_time = time.time()

        # xcodec2 を RAM に退避
        self.xcodec2.move_to_ram()
        self.whisper.load_model(transcript_model)

        result = self.whisper.audio2text(audio_file_name)

        self.whisper.unload()
        print(f"Whisper model inference time : {time.time()-old_time:.1f} sec")

        return result