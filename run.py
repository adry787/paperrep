#!/usr/bin/env python3
import torch
from models import MiniTransformer, MiniResNet

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev}")

m = MiniTransformer().to(dev)
print(f"Transformer: {sum(p.numel() for p in m.parameters()):,} params")

m = MiniResNet().to(dev)
print(f"ResNet: {sum(p.numel() for p in m.parameters()):,} params")
