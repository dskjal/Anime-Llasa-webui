from xcodec2.modeling_xcodec2 import XCodec2Model
from xcodec2.configuration_bigcodec import BigCodecConfig
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import soundfile as sf
import librosa
import torch
import time
import numpy as np
import os
from utils import cpu_device, cuda_device, has_available_vram_gb

class XCodec2Wrapper():
    def __init__(self, use_fp16:bool=True, use_44khz:bool=True):
        self.load_model(use_44khz)
        self.quantize(use_fp16)
        self.vram_gb = 2

    def get_required_vram_size_gb(self) -> float:
        return self.vram_gb
    
    def load_model(self, use_44khz:bool=True):
        self.use_44khz = use_44khz
        self.xcodec2_model_name = "HKUSTAudio/xcodec2"
        self.encoder_sr = 16000   # xcodec2 の動作周波数
        self.decoder_sr = self.encoder_sr

        old_time = time.time()
        if use_44khz:
            self.xcodec2_model_name = "NandemoGHS/Anime-XCodec2-44.1kHz"
            self.decoder_sr = 44100
            ckpt_path = hf_hub_download(repo_id=self.xcodec2_model_name, filename="model.safetensors")
            ckpt = {}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    ckpt[k.replace(".beta", ".bias")] = f.get_tensor(k)
                codec_config = BigCodecConfig.from_pretrained(self.xcodec2_model_name)
                self.Codec_model = XCodec2Model.from_pretrained(
                    None, config=codec_config, state_dict=ckpt
                )
        else:
            self.Codec_model = XCodec2Model.from_pretrained(self.xcodec2_model_name)
            
        self.Codec_model.eval()
        print(f"XCodec2 load time : {time.time()-old_time:.1f} sec")

    def quantize(self, use_fp16:bool=True):
        self.use_fp16 = use_fp16
        self.dtype = torch.float32
        old_time = time.time()
        if use_fp16:
            self.dtype = torch.float16
            # fp16 に変換
            self.Codec_model = self.Codec_model.half()
            # LayerNorm モジュールを特定し、FP32 に戻す
            def cast_ln_to_fp32(m):
                # PyTorchの LayerNorm のインスタンスであれば
                if isinstance(m, torch.nn.LayerNorm):
                    m.float() # LayerNorm の重みを FP32 にキャスト
                    
            # モデル全体に適用
            self.Codec_model.apply(cast_ln_to_fp32)
        print(f"XCodec2 quantize time : {time.time()-old_time:.1f} sec")

    def move_to_ram(self) -> None:
        self.Codec_model.to(cpu_device)
        torch.cuda.empty_cache()

    def move_to_vram(self) -> None:
        if self.Codec_model.device == cuda_device:
            return
        
        required_vram_gb = 2 * 1024**3 if self.use_fp16 else 3.5 * 1024**3
        if not has_available_vram_gb(required_vram_gb) :
            print("Load XCodec2 to vram failed. CPU is used.")
            return
        
        self.Codec_model.to(cuda_device)

    def speech2token(self, audio_file:str) -> np.array:
        if not os.path.isfile(audio_file):
            raise FileNotFoundError(audio_file)
        
        self.move_to_vram()

        wav, _ = librosa.load(audio_file, sr=self.encoder_sr)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0) # Shape: (1, T)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                return self.Codec_model.encode_code(input_waveform=wav_tensor).cpu()[0, 0, :].numpy()
        
    def token2speech(self, tokens:list, audio_file_path:str) -> str:
        self.move_to_vram()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                speech_tokens = torch.tensor(tokens, device=cuda_device).unsqueeze(0).unsqueeze(0)
                gen_wav = self.Codec_model.decode_code(speech_tokens)

        sf.write(audio_file_path, gen_wav[0, 0, :].cpu().numpy(), self.decoder_sr)
        # with open(f"./outputs/{filename}.txt", "w", encoding="utf-8") as f:
        #     f.write(','.join([str(i) for i in tokens]))
