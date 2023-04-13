from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from littlebird.distance_matracies import BidirectionalALiBi, UniformDistanceMatrix

def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
    """Fast nd matrix multiplication with transpose"""
    # faster replacement of torch.einsum (bhqd,bhkd->bhqk)
    return torch.bmm(
        inp_1.reshape((-1,) + inp_1.shape[-2:]),
        inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2),
    ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))


def torch_bmm_nd(inp_1, inp_2, ndim=None):
    """Fast nd matrix multiplication"""
    # faster replacement of torch.einsum ("bhqk,bhkd->bhqd")
    return torch.bmm(
        inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])
    ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1]))


def transpose_for_scores(x, num_attn_head, attn_head_size):
    new_x_shape = x.size()[:-1] + (num_attn_head, attn_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


class PackAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(PackAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True
        )

    def forward(self, P: torch.Tensor, X: torch.Tensor, attention_mask: torch.Tensor):
        attn, _ = self.mha(P, X, X, attention_mask)
        return attn


class UnpackSlidingWindowAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pack_len: int,
        dim: int,
        num_attention_heads: int = 8,
        block_size=64,
    ) -> None:
        super(UnpackSlidingWindowAttention, self).__init__()

        self.attn_head_size = int(dim / num_attention_heads)
        self.num_attention_heads = num_attention_heads
        self.block_size = block_size
        self.seq_len = seq_len
        self.pack_len = pack_len

        self.Q = nn.Linear(dim, self.attn_head_size * num_attention_heads)
        self.K = nn.Linear(dim, self.attn_head_size * num_attention_heads)
        self.V = nn.Linear(dim, self.attn_head_size * num_attention_heads)

        self.bialibi = BidirectionalALiBi(
            self.num_attention_heads, self.seq_len
        )
        self.uniform_dist_mat = UniformDistanceMatrix(
            self.num_attention_heads, self.block_size, self.seq_len, self.pack_len
        )

        # These are the indicies AFTER transposing - note the row and column should be the same
        self.register_buffer("middle_band_distance_indicies", None, persistent=False)

        # Add masks to module state
        self.register_buffer("mask_block", None, persistent=False)
        self.register_buffer("middle_band_mask", None, persistent=False)
        self.register_buffer("mask_k", None, persistent=False)
        self.register_buffer("mask_v", None, persistent=False)

    def get_middle_band_distances(self, distance_matrix):

        if self.middle_band_distance_indicies is None:
            self.middle_band_indicies = []

            row = 0
            # from 2 to -1 block
            for i in range(
                self.block_size * 2,
                self.seq_len - self.block_size,
            ):
                offset = ((i // self.block_size) - 2) + 1
                self.middle_band_indicies += [[row, i] for i in range(self.block_size)]

                for j in range(
                    self.block_size * offset, self.block_size * (offset + 3)
                ):
                    self.middle_band_indicies.append([row, j])
                row += 1
            self.middle_band_indicies = torch.as_tensor(self.middle_band_indicies)

        return distance_matrix[
            :, self.middle_band_indicies[:, 0], self.middle_band_indicies[:, 1]
        ].view(
            -1,
            (self.seq_len // self.block_size) - 3,
            self.block_size,
            self.block_size * 4,
        )

    def create_masks_for_block_sparse_attn(
        self, attention_mask: torch.Tensor, block_size: int
    ):
        if self.mask_block is None:
            batch_size, seq_length = attention_mask.size()

            def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
                exp_blocked_to_pad = torch.cat(
                    [
                        to_blocked_mask[:, 1:-2],
                        to_blocked_mask[:, 2:-1],
                        to_blocked_mask[:, 3:],
                    ],
                    dim=2,
                )
                band_mask = torch.einsum(
                    "blq,blk->blqk", from_blocked_mask[:, 2:-1], exp_blocked_to_pad
                )
                band_mask.unsqueeze_(1)
                return band_mask

            mask_block = attention_mask.view(
                batch_size, seq_length // block_size, block_size
            ) # bsz, block_cnt, block_size
            band_mask = create_band_mask_from_inputs(
                mask_block, mask_block
            ) 

            mask_v = attention_mask.view(batch_size, 1, seq_length, 1)
            mask_k = attention_mask.view(batch_size, 1, 1, seq_length)

            self.mask_block, self.band_mask, self.mask_v, self.mask_k = mask_block.float(), band_mask.float(), mask_v.float(), mask_k.float()

    def forward(
        self,
        X: torch.Tensor,
        Cp: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        attn_penalty = -10000.0

        if self.mask_block is None:
            self.create_masks_for_block_sparse_attn(attention_mask, self.block_size)
        
        D = self.bialibi().permute(0, 2, 1)
        Dp = self.uniform_dist_mat().permute(0, 2, 1)

        rsqrt_d = 1 / math.sqrt(self.attn_head_size)

        batch_size, seq_len, dim = X.shape

        query = transpose_for_scores(
            self.Q(X), self.num_attention_heads, self.attn_head_size
        )  # bsz, head, seq_len, head_dim
        key_x = transpose_for_scores(
            self.K(X), self.num_attention_heads, self.attn_head_size
        )  # bsz, head, seq_len, head_dim
        value_x = transpose_for_scores(
            self.V(X), self.num_attention_heads, self.attn_head_size
        )  # bsz, head, seq_len, head_dim

        key_Cp = transpose_for_scores(
            self.K(Cp), self.num_attention_heads, self.attn_head_size
        )  # bsz, head, pack_len, head_dim
        value_Cp = transpose_for_scores(
            self.V(Cp), self.num_attention_heads, self.attn_head_size
        )  # bsz, head, pack_len, head_dim

        # Step 1 calculate the attention scores for the packed data
        key_cp_attn = torch.matmul(query, key_Cp.transpose(2, 3))
        key_cp_attn = key_cp_attn * rsqrt_d
        key_cp_attn[:] -= Dp
        key_cp_attn += (1.0 - self.mask_v) * attn_penalty
        key_cp_attn = F.softmax(key_cp_attn, dim=-1)  # bsz, heads, seq_len, pack_len

        packed_context = torch_bmm_nd(key_cp_attn, value_Cp, ndim=4) * self.mask_v

        # Step 2. Calculate attention scores for the first row of sliding window-global
        query_block = query.view(
            batch_size,
            self.num_attention_heads,
            seq_len // self.block_size,
            self.block_size,
            -1,
        )
        key_x_block = key_x.view(
            batch_size,
            self.num_attention_heads,
            seq_len // self.block_size,
            self.block_size,
            -1,
        )
        value_x_block = value_x.view(
            batch_size,
            self.num_attention_heads,
            seq_len // self.block_size,
            self.block_size,
            -1,
        )

        # Step 2.1. process the first two rows
        first_two_rows_key_matrix = torch.cat(
            [
                key_x_block[:, :, 0],
                key_x_block[:, :, 1],
                key_x_block[:, :, 2],
                key_x_block[:, :, 3],
            ],
            dim=2,
        )
        first_two_rows_value_matrix = torch.cat(
            [
                value_x_block[:, :, 0],
                value_x_block[:, :, 1],
                value_x_block[:, :, 2],
                value_x_block[:, :, 3],
            ],
            dim=2,
        )
        first_two_query_blocks = torch.cat(
            [query_block[:, :, 0], query_block[:, :, 1]], dim=2
        )

        first_two_rows_attn = torch_bmm_nd_transpose(
            first_two_query_blocks, first_two_rows_key_matrix, ndim=4
        )
        first_two_rows_attn *= rsqrt_d
        first_two_rows_attn -= D[:, : self.block_size * 2, : self.block_size * 4]
        first_two_rows_attn += (1.0 - self.mask_v[:, :, :self.block_size * 2, :self.block_size * 4]) * attn_penalty
        first_two_rows_attn = F.softmax(first_two_rows_attn, dim=-1)

        first_two_rows_context = torch_bmm_nd(
            first_two_rows_attn, first_two_rows_value_matrix, ndim=4
        )
        _, __, ftr_3d, ftr_4d = first_two_rows_context.shape
        first_two_rows_context = first_two_rows_context.view(
            batch_size, self.num_attention_heads, 2, ftr_3d // 2, ftr_4d
        )  # bsz, heads, 2(blocks), block_size, block_size*4

        # step 2.2 calculate the middle part of the matrix
        # the trick described in the bigbird paper is used

        middle_band_key_matrix = torch.cat(
            [
                key_x_block[:, :, 1:-2],  # roll back one
                key_x_block[:, :, 2:-1],
                key_x_block[:, :, 3:],  # roll forward one
            ],
            dim=3,
        )
        middle_band_value_matrix = torch.cat(
            [
                value_x_block[:, :, 1:-2],  # roll back one
                value_x_block[:, :, 2:-1],
                value_x_block[:, :, 3:],  # roll forward one
            ],
            dim=3,
        )



        # get the diagnol
        middle_band_sliding = torch_bmm_nd_transpose(
            query_block[:, :, 2:-1], middle_band_key_matrix, ndim=5
        )
        middle_band_sliding += (1.0 - self.band_mask) * attn_penalty

        # get the global
        middle_band_global = torch.einsum(
            "bhlqd,bhkd->bhlqk", query_block[:, :, 2:-1], key_x_block[:, :, 0]
        )
        middle_band_global += (1.0 - self.mask_block[:,2:-1,:].unsqueeze(3)) * attn_penalty

        middle_band_attn = torch.cat([middle_band_global, middle_band_sliding], dim=-1)
        middle_band_attn *= rsqrt_d
        middle_band_attn -= self.get_middle_band_distances(D)
        middle_band_attn = F.softmax(middle_band_attn, dim=-1)

        middle_band_context = torch.einsum(
            "bhlqk,bhkd->bhlqd",
            middle_band_attn[:, :, :, :, : self.block_size],
            value_x_block[:, :, 0],
        )
        middle_band_context += torch_bmm_nd(
            middle_band_attn[:, :, :, :, self.block_size : 4 * self.block_size],
            middle_band_value_matrix,
            ndim=5,
        )

        # step 2.3
        # calcualte the last row
        last_row_key_matrix = torch.cat(
            [
                key_x_block[:, :, 0],
                key_x_block[:, :, -3],
                key_x_block[:, :, -2],
                key_x_block[:, :, -1],
            ],
            dim=2,
        )
        last_row_value_matrix = torch.cat(
            [
                value_x_block[:, :, 0],
                value_x_block[:, :, -3],
                value_x_block[:, :, -2],
                value_x_block[:, :, -1],
            ],
            dim=2,
        )

        last_row_attn = torch_bmm_nd_transpose(
            query_block[:, :, -1], last_row_key_matrix, ndim=4
        )
        last_row_attn *= rsqrt_d
        last_row_attn -= D[:, -self.block_size :, -self.block_size * 4 :]
        last_row_attn = F.softmax(last_row_attn, dim=-1)

        last_row_context = torch_bmm_nd(last_row_attn, last_row_value_matrix, ndim=4)
        last_row_context.unsqueeze_(2)

        context_layer = torch.cat(
            [first_two_rows_context, middle_band_context, last_row_context], dim=2
        )
        context_layer = context_layer.view(
            (batch_size, self.num_attention_heads, seq_len, -1)
        )

        Cx = context_layer + packed_context
        Cx = Cx.view(
            batch_size, seq_len, self.num_attention_heads * self.attn_head_size
        ) * self.mask_v.squeeze(1)

        return Cx

        # query_layer = self.transpose_for_scores(self.query(hidden_states))
        # key_layer = self.transpose_for_scores(self.key(hidden_states))
        # value_layer = self.transpose_for_scores(self.value(hidden_states))
