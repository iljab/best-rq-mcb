import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

class RandomProjectionQuantizer(nn.Module):

    def __init__(self, input_dim, cb_dim, cb_vocab):
        super().__init__()

        self.input_dim = input_dim
        self.cb_dim = cb_dim
        self.cb_vocab = cb_vocab

        # Section 3.1 "projection matrix A use Xavier initialization"
        P_init = torch.empty((input_dim, cb_dim))
        self.register_buffer("P", nn.init.xavier_uniform_(P_init))

        # normalize random matrix for codebook
        self.register_buffer("CB", F.normalize(torch.randn(cb_vocab, cb_dim)))

    def forward(self, x):
        x_proj = x @ self.P
        x_proj = F.normalize(x_proj, dim=1)

        cosine_similarities = F.cosine_similarity(x_proj.unsqueeze(1), self.CB.unsqueeze(1), dim=-1)
        closest_indices = cosine_similarities.argmax(dim=1)

        return closest_indices
