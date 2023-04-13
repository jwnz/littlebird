import torch
import torch.nn as nn

from littlebird.littlebird_layer import LittleBirdLayer


class LittleBirdModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pack_len: int,
        vocab_size: int,
        d_model: int,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        d_ff: int = 2048,
        dropout_p: float = 0.1,
        block_size: int = 64
    ):
        super(LittleBirdModel, self).__init__()
        self.d_model = d_model

        self.projected_embeddings = nn.Parameter(
            torch.Tensor(pack_len, self.d_model)
        )
        nn.init.normal_(self.projected_embeddings, mean=0.0, std=self.d_model**-0.5)

        self.embedding = nn.Embedding(
            vocab_size, d_model
        )  # load embedding from pretrained model?
        self.dropout = nn.Dropout(p=dropout_p)

        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList(
            [
                LittleBirdLayer(
                    seq_len,
                    pack_len,
                    d_model=d_model,
                    num_attention_heads=num_attention_heads,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                    block_size=block_size
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:

        batch_size, seq_length = inputs.size()

        # embed inputs
        embedded = self.embedding(inputs)

        # expand the P embeddings to batch size
        seq_length, dim = self.projected_embeddings.size()
        projected_embedded = self.projected_embeddings.unsqueeze(0).expand(
            batch_size, seq_length, dim
        )

        X = self.dropout(embedded)
        P = self.dropout(projected_embedded)

        for layer in self.layers:
            P, X = layer(P, X, attention_mask)

        return X
