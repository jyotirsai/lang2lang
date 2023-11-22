import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, embed_size)
        return self.embedding(x) * math.sqrt(self.embed_size)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, seq_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )  # (embed_size / 2)
        pe = torch.zeros(1, seq_len, embed_size)  # (1, seq_len, embed_size)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # (batch, seq_len, embed_size)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.Wq = nn.Linear(embed_size, embed_size)
        self.Wk = nn.Linear(embed_size, embed_size)
        self.Wv = nn.Linear(embed_size, embed_size)
        self.Wo = nn.Linear(embed_size, embed_size)

        self.head_size = embed_size // num_heads

    def attention(self, q, k, v, mask):
        # q @ k then scale
        # (batch, num_heads, seq_len, head_size) --> (batch, num_heads, seq_len, seq_len)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention = F.softmax(attention, dim=-1)

        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_size) --> (batch, num_heads, seq_len, head_size)
        return self.dropout(attention @ v)

    def forward(self, q, k, v, mask):
        # (batch, seq_len, embed_size) --> (batch, seq_len, embed_size)
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # (batch, seq_len, embed_size) --> (batch, seq_len, num_heads, head_size) --> (batch, num_heads, seq_len, head_size)
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.head_size).transpose(
            1, 2
        )
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.head_size).transpose(
            1, 2
        )
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.head_size).transpose(
            1, 2
        )

        # (batch, num_heads, seq_len, head_size)
        x = self.attention(q, k, v, mask)

        # (batch, num_heads, seq_len, head_size) --> (batch, seq_len, embed_size)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .reshape(x.shape[0], x.shape[2], self.head_size * self.num_heads)
        )

        return self.Wo(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()
        self.lin_1 = nn.Linear(embed_size, hidden_size)
        self.lin_2 = nn.Linear(hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin_2(self.dropout(torch.relu(self.lin_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(embed_size, hidden_size, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(2)])

    def forward(self, x, mask):
        x = self.layer_norm[0](x + self.multi_head_attention(x, x, x, mask))
        x = self.layer_norm[1](x + self.feed_forward(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(
            embed_size, num_heads, dropout
        )
        self.multi_head_attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(embed_size, hidden_size, dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(3)])

    def forward(self, x, encoder_output, padding_mask, lookahead_mask):
        x = self.layer_norm[0](
            x + self.masked_multi_head_attention(x, x, x, lookahead_mask)
        )
        x = self.layer_norm[1](
            x
            + self.multi_head_attention(x, encoder_output, encoder_output, padding_mask)
        )
        x = self.layer_norm[2](x + self.feed_forward(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embed_size,
        seq_len,
        num_heads,
        hidden_size,
        dropout,
        num_layers,
    ):
        super().__init__()
        self.input_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.output_embedding = nn.Embedding(output_vocab_size, embed_size)
        self.input_positional_encoding = PositionalEncoding(
            embed_size, seq_len, dropout
        )
        self.output_positional_encoding = PositionalEncoding(
            embed_size, seq_len, dropout
        )
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(embed_size, num_heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(embed_size, num_heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(embed_size, output_vocab_size)

    def encode(self, x, padding_mask):
        x = self.input_embedding(x)
        x = self.input_positional_encoding(x)

        for encoder_layer in self.encoder:
            x = encoder_layer(x, padding_mask)

        return x

    def decode(self, x, encoder_output, padding_mask, lookahead_mask):
        x = self.output_embedding(x)
        x = self.output_positional_encoding(x)
        for decoder_layer in self.decoder:
            x = decoder_layer(x, encoder_output, padding_mask, lookahead_mask)

        return x

    def predict(self, x):
        return self.linear(x)
