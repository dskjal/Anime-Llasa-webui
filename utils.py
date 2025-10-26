import torch

cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else cpu_device

def has_available_vram_gb(required_vram_gb):
    free, _ = torch.cuda.mem_get_info(cuda_device)
    return free >= required_vram_gb