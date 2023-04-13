import torch
import torch.nn as nn


class BidirectionalALiBi(nn.Module):
    def __init__(self, num_attention_heads: int, seq_len: int):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.seq_len = seq_len

        self.alpha = nn.Parameter(
            torch.empty(self.num_attention_heads), requires_grad=True
        )
        self.beta = nn.Parameter(
            torch.empty(self.num_attention_heads), requires_grad=True
        )
        self.gamma = nn.Parameter(
            torch.empty(self.num_attention_heads), requires_grad=True
        )

        # initialize weights from the uniform dist
        nn.init.uniform_(self.alpha)
        nn.init.uniform_(self.beta)
        nn.init.uniform_(self.gamma)

        self.register_buffer("distances", None, persistent=False)

    def get_distances(self):
        row_i = torch.arange(self.seq_len, dtype=torch.float32)
        col_i = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(-1)
        distances = (row_i - col_i).abs()

        return distances.unsqueeze(0).expand(self.num_attention_heads, -1, -1).clone()

    def forward(self):
        if self.distances is None:
            self.distances = self.get_distances()

        # step 1: get gamma mask
        gamma_mask = torch.triu(torch.ones_like(self.distances), diagonal=1)
        gamma_mask *= self.gamma.view(-1, 1, 1)

        # step 2: get beta mask
        beta_mask = torch.tril(torch.ones_like(self.distances), diagonal=-1)
        beta_mask *= self.beta.view(-1, 1, 1)

        # step 3: combine beta and gamma masks
        mask = beta_mask + gamma_mask

        # step 4: set the alphas
        mask[:, 0, :] = 1.0
        mask[:, :, 0] = 1.0
        mask[:, 1:, 0] *= self.alpha.unsqueeze(1)
        mask[:, 0, 1:] *= self.alpha.unsqueeze(1)
        mask[:, 0, 0] *= 0.0

        return self.distances * mask


class UniformDistanceMatrix(nn.Module):
    def __init__(
        self, num_attention_heads: int, row_cnt: int, col_cnt: int, block_size: int
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.row_cnt = row_cnt
        self.col_cnt = col_cnt
        self.block_size = block_size

        self.beta = nn.Parameter(
            torch.empty(self.num_attention_heads), requires_grad=True
        )
        self.gamma = nn.Parameter(
            torch.empty(self.num_attention_heads), requires_grad=True
        )

        # initialize weights from the uniform dist
        nn.init.uniform_(self.beta)
        nn.init.uniform_(self.gamma)

        self.register_buffer("distances", None, persistent=False)

    def get_distances(self):
        return torch.ones(self.num_attention_heads, self.row_cnt, self.col_cnt)

    def forward(self):
        if self.distances is None:
            self.distances = self.get_distances()

        mask = ((self.beta + self.gamma) / 2) * self.block_size

        return self.distances * mask.view(-1, 1, 1)
