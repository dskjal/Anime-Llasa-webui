import torch
import soundfile as sf
import librosa

from xcodec2.modeling_xcodec2 import XCodec2Model

cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"

USE_FP16 = True
DTYPE = torch.float16 if USE_FP16 else torch.float32

USE_44KHZ = True
XCODEC2_MODEL = "HKUSTAudio/xcodec2"
ENCODER_SR = 16000   # xcodec2 の動作周波数
DECODER_SR = ENCODER_SR
if USE_44KHZ:
    XCODEC2_MODEL = "NandemoGHS/Anime-XCodec2-44.1kHz"
    DECODER_SR = 44100


model = XCodec2Model.from_pretrained(XCODEC2_MODEL)#, local_files_only=True)
model.eval()

if USE_FP16:
    model = model.half()
    # LayerNorm モジュールを特定し、FP32 に戻す
    def cast_ln_to_fp32(m):
        # PyTorchの LayerNorm のインスタンスであれば
        if isinstance(m, torch.nn.LayerNorm):
            # LayerNorm の重みを FP32 にキャスト
            m.float()
            
    # モデル全体に適用
    model.apply(cast_ln_to_fp32)

model.to(cuda_device)

wav, sr = librosa.load("test.wav", sr=ENCODER_SR)
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0) # Shape: (1, T)
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=DTYPE):
    # Only 16khz speech
        vq_code = model.encode_code(input_waveform=wav_tensor)
        #print("Code:", vq_code )  
        recon_wav = model.decode_code(vq_code).cpu()       # Shape: (1, 1, T')

sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), DECODER_SR)
print("Done! Check reconstructed.wav")