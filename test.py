import sys, torch
print(sys.executable)
print("torch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())  # saldrá False y está bien
