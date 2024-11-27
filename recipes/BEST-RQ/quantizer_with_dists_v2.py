import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.linalg import vector_norm

from einops import rearrange, repeat, pack, unpack

def exists(val):
    return val is not None

class RandomProjectionQuantizerV2(nn.Module):

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        codebook_dim,
        num_codebooks = 1,
        return_distances = True,
        **kwargs
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.return_distances = return_distances

        rand_projs = torch.empty(num_codebooks, dim, codebook_dim)
        nn.init.xavier_normal_(rand_projs)

        self.register_buffer('rand_projs', rand_projs)
        self.register_buffer("CB", F.normalize(torch.randn(num_codebooks, codebook_size, codebook_dim)))

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        x = self.norm(x)
        x = einsum('b t d, h d k -> h b t k', x, self.rand_projs)

        indices = []
        distances = []
        for h in range(self.num_codebooks):
            distances_cossim = F.cosine_similarity(self.CB[h].unsqueeze(1), x[h].unsqueeze(1), dim=-1)
            distances.append(distances_cossim)

            indices_cossim = distances_cossim.argmax(dim=1)
            indices.append(indices_cossim)

        if self.return_distances:
            return torch.stack(indices).permute(1, 2, 0), torch.stack(distances).permute(0, 1, 3, 2)
        return torch.stack(indices).permute(1, 2, 0)