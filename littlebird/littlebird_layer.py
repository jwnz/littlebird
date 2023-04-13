import torch
import torch.nn as nn

from littlebird.attention import PackAttention
from littlebird.attention import UnpackSlidingWindowAttention
from littlebird.feed_forward import PositionwiseFeedForwardNetwork


class LittleBirdLayer(nn.Module):
    """
    Implementation of LittleBird: Efficient Faster & Longer Transformer for Question Answering (https://arxiv.org/abs/2106.01540.pdf)
    """

    def __init__(
        self,
        seq_len: int,
        pack_len: int,
        d_model: int = 512,
        num_attention_heads: int = 8,
        d_ff: int = 2048,
        dropout_p: float = 0.3,
        block_size: int = 64,
    ) -> None:
        super(LittleBirdLayer, self).__init__()

        self.pack_attn = PackAttention(d_model, num_attention_heads)
        self.unpack_sliding_attn = UnpackSlidingWindowAttention(
            seq_len, pack_len, d_model, num_attention_heads, block_size
        )
        self.feed_forward = PositionwiseFeedForwardNetwork(d_model, d_ff, dropout_p)

        self.pack_attn_layer_norm = nn.LayerNorm(d_model)
        self.unpack_sliding_attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        P: torch.FloatTensor,
        X: torch.FloatTensor,
        attention_mask: torch.FloatTensor = None,
    ):
        Cp = self.pack_attn(P, X, attention_mask)
        P0 = self.pack_attn_layer_norm(Cp + P)

        Cx = self.unpack_sliding_attn(X, Cp, attention_mask)
        A = self.unpack_sliding_attn_layer_norm(Cx + X)

        X0 = self.ffn_layer_norm(self.feed_forward(A) + A)

        return P0, X0
