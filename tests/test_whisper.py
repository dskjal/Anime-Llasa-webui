import torch
import librosa
from transformers import pipeline

whisper_turbo_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=torch.device("cuda:0"))

wav, _ = librosa.load("./test.wav", sr=16000)
prompt_text = whisper_turbo_pipe(wav)['text'].strip()
print(prompt_text)