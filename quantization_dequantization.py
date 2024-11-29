import torch 
import numpy as np
# Function to quantize embeddings to int8
def quantization(embeddings: torch.Tensor, bits=8) -> torch.Tensor:
    quantized_embeddings=np.round(embeddings.cpu() * (2**(bits - 1)- 1)).to(torch.int8)
    return quantized_embeddings


# Function to dequantize int8 embeddings back to float32
def dequantization(quantized_embeddings: torch.Tensor, bits=8) -> torch.Tensor:
    dequantized_embeddings = quantized_embeddings / (2 ** (bits -1)-1)
    return dequantized_embeddings