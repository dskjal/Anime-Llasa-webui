import torch
import librosa
import gc
import time
from transformers import pipeline
from utils import cpu_device, cuda_device, has_available_vram_gb

ANIME_WHISPER = "litagin/anime-whisper"
WHISPER_LARGE_V3_TURBO = "openai/whisper-large-v3-turbo"

class Whisper():
    """
    self.pipe.model.to では一部のウェイトしか移動できないので、使用時に動的にロードする。使用後はアンロードを忘れないようにする。
    """
    def __init__(self, audio_sr=16000) -> None:
        self.pipe = None
        self.audio_sr = audio_sr
        self.generate_kwargs = {
            "language": "Japanese",
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        }

    def get_loaded_model(self) -> str:
        """
        ロードされているときはモデル名を返し、ロードされていないときは空文字列を返す。
        """
        if self.pipe:
            return ANIME_WHISPER if self.use_anime_whisper else WHISPER_LARGE_V3_TURBO
        return ""
    
    def load_model(self, use_anime_whisper=True) -> None:
        self.use_anime_whisper = use_anime_whisper
        model_name = ANIME_WHISPER if use_anime_whisper else WHISPER_LARGE_V3_TURBO
        old_time = time.time()
        if use_anime_whisper:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=cuda_device,
                torch_dtype=torch.float16,
                chunk_length_s=30.0,
                batch_size=64,
            )
        else:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=cuda_device,
                torch_dtype=torch.float16)
        self.pipe.model.eval()
        print(f"{model_name} load time : {time.time()-old_time:.1f} sec")
            

    def unload(self) -> None:
        if self.pipe:
            del self.pipe
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # def move_to_ram(self) -> None:
    #     self.pipe.model.to(cpu_device)

    # def move_to_vram(self) -> None:
    #     if self.pipe.device == cuda_device:
    #         return
        
    #     required_vram = 2 * 1024**3
    #     if has_available_vram_gb(required_vram) :
    #         self.pipe.model.to(cuda_device)
    #         return
        
    #     print("Load Whisper model to vram failed. CPU is used.")

    def audio2text(self, audio_path):
        wav, _ = librosa.load(audio_path, sr=self.audio_sr)
        if self.use_anime_whisper:
            text = self.pipe(wav, generate_kwargs=self.generate_kwargs)['text'].strip()
        else:
            text = self.pipe(wav)['text'].strip()

        return text