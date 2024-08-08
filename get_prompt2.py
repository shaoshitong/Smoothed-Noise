
import torch.fft
import torch
x = 8
t = torch.arange(x)
result = torch.fft.fft(t)
print(result)
print(torch.fft.fftshift(result))