import torch

checkpoint = torch.load("mobile-actions_q8_ekv1024.litertlm", map_location="cpu", weights_only=False)
print(f"타입: {type(checkpoint)}")
print(f"키 목록: {list(checkpoint.keys())[:10]}")