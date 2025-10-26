from llama_cpp import Llama
import torch

import logging
log_file = "verbose_output.txt"

MAX_TOKENS = 1024 
MAX_CONTEXT_SIZE = 4096
MAX_TOKENS = min(MAX_TOKENS, MAX_CONTEXT_SIZE)
N_GPU_LAYERS = -1 if torch.cuda.is_available() else 0 

GGUF_MODEL_PATH = "../models/Anime-Llasa-3B-Captions.Q8_0.gguf"
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=MAX_CONTEXT_SIZE,
    #chat_format="llama-3",
    verbose=False,
)

input_text = 'Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, intending to safeguard some from the harsh truths. One day, I hope you understand the reasons behind my actions. Until then, Anna, please, bear with me.'

with torch.no_grad():
    prompt = f"Convert the text to speech:{input_text}"

    llm.reset()
    prompt_tokens = llm.tokenize(prompt.encode('utf-8'))
    llm.eval(prompt_tokens)
    generated_tokens = []
    outputs = ""

    for i in range(MAX_TOKENS):  # max_tokens
        # 次のトークンをサンプリング
        token = llm.sample(
            top_k=40,
            top_p=0.95,
            temp=0.9,
        )

        TOKEN_OFFSET = 128264
        if token >= TOKEN_OFFSET:
            generated_tokens.append(token)
            print(token-TOKEN_OFFSET)

        # トークン追加
        llm.eval([token])

        # 停止条件
        if token == llm.token_eos():
            print("\n[EOS token reached]")
            break

#llm._sampler.close()
llm.close()
