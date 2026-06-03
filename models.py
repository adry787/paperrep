#!/usr/bin/env python3
import torch, torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, v=1000, d=128, h=4, l=3):
        super().__init__()
        self.emb = nn.Embedding(v, d)
        self.pos = nn.Embedding(512, d)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, batch_first=True), l)
        self.head = nn.Linear(d, v)

    def forward(self, x):
        b, s = x.shape
        p = torch.arange(s, device=x.device).unsqueeze(0)
        return self.head(self.tf(self.emb(x) + self.pos(p)))


class MiniResNet(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, n)
        )

    def forward(self, x):
        return self.net(x)
