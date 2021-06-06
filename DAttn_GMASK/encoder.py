from functools import partial
import math
from typing import Any, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(
        self,
        args,
        text_field,
        bidirectional: bool = True,
        layer_norm: bool = False,
        highway_bias: float = 0.0,
        pooling: str = "average",
        embedding_dropout: float = 0.1,
        rescale: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """Constructs an model to compute embeddings."""
        super(Embedder, self).__init__()

        # Save values
        self.args = args
        self.device = device
        pad_index = text_field.pad_index()
        self.pad_index = pad_index
        self.embdrop = nn.Dropout(embedding_dropout)
        self.pooling = pooling

        num_embeddings = len(text_field.vocabulary)
        self.num_embeddings = num_embeddings
        if args.small_data:
            self.embedding_size = 300
            print("random initializing for debugging")
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        else:
            print(f'Loading embeddings from "{args.embedding_path}"')
            embedding_matrix = text_field.load_embeddings(args.embedding_path)
            self.embedding_size = embedding_matrix.size(1)
            # Create models/parameters
            self.embedding = nn.Embedding(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_size,
                padding_idx=self.pad_index,
            )
            self.embedding.weight.data = embedding_matrix
        self.embedding.weight.requires_grad = False

        self.input_linear = nn.Linear(self.embedding_size, self.args.ffn_hidden_size, bias=False)  # linear transformation
        self.output_size = self.args.ffn_hidden_size

    def forward(self, data):
        # Embed
        embedded = self.embdrop(self.embedding(data))
        output = self.input_linear(embedded)
        return output
