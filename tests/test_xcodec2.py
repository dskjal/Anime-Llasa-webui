import torch
import soundfile as sf
import librosa

from xcodec2.modeling_xcodec2 import XCodec2Model

cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000   # xcodec2 の動作限界


model_path = "HKUSTAudio/xcodec2"
model = XCodec2Model.from_pretrained(model_path)#, local_files_only=True)
model.eval().to(cuda_device)

#wav, sr = sf.read("test16khz.wav")
wav, sr = librosa.load("test.wav", sr=TARGET_SR)
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0) # Shape: (1, T)
with torch.no_grad():
   # Only 16khz speech
    vq_code = model.encode_code(input_waveform=wav_tensor)
    #print("Code:", vq_code )  
    recon_wav = model.decode_code(vq_code).cpu()       # Shape: (1, 1, T')

sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
print("Done! Check reconstructed.wav")