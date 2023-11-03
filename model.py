import torch
import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(
            self.d_model
        )  # (batch, seq_len, d_model), first two dims are from x, last is from embedding


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1) , id representing position in sequence
        divisor = torch.exp(
            (torch.arange(0, d_model, 2).float() / d_model) * -math.log(10000.0)
        )  # ([seq_len/2])

        # position * divisor --> (seq_len, seq_len / 2)
        pe[:, 0::2] = torch.sin(
            position * divisor
        )  # apply sin to all even columns of pe
        pe[:, 1::2] = torch.cos(
            position * divisor
        )  # apply cos to all odd columns of pe

        # need to add to input embedding which is (batch, seq_len, d_model)
        # convert 2d matrix into 3d so we can add properly
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)  # save pe

    def forward(self, x):
        # x is the embedding matrix
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # seq_len in batch may be less than seq_len specified, so adjust accordingly
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dk = d_model // h
        self.dropout = nn.Dropout(dropout)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, head, d_model) --> (batch, head, seq_len, d_model)
        return x.view(x.shape[0], x.shape[1], self.h, self.dk).permute(0, 2, 1, 3)

    def attention(self, q, k, v, mask, dropout):
        scores = q @ k.transpose(2, 3) // math.sqrt(self.dk)
        if mask is not None:
            scores.masked_fill(mask == 0, -1e9)

        attention_score = torch.softmax(
            scores, dim=-1
        )  # (batch, head, seq_len, seq_len)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return attention_score @ v  # (batch, head, seq_len, dk)

    def forward(self, q, k, v, mask):
        query = self.Wq(q)  # (batch, seq_len, d_model)
        key = self.Wk(k)
        value = self.Wv(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, head, d_model) --> (batch, head, seq_len, d_model)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        out = self.attention(query, key, value, mask, self.dropout)

        # (batch, head, seq_len, d_model) --> (batch, seq_len, head, d_model) --> (batch, seq_len, d_model)
        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .view(out.shape[0], -1, self.h * self.dk)
        )

        return self.Wo(out)  # (batch, seq_len, d_model)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.lin2(self.dropout(torch.relu(self.lin1(x))))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.lnorm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.lnorm[0](
            x + self.dropout(self.self_attention_block(x, x, x, src_mask))
        )
        x = self.lnorm[1](x + self.dropout(self.feed_forward_block(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.lnorm = nn.LayerNorm(d_model)
        self.layers = layers

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.lnorm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.lnorm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.lnorm[0](
            x + self.dropout(self.self_attention_block(x, x, x, tgt_mask))
        )
        x = self.lnorm[1](
            x + self.dropout(self.cross_attention_block(x, enc_out, enc_out, src_mask))
        )
        x = self.lnorm[2](x + self.feed_forward_block(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int):
        super().__init__()
        self.lnorm = nn.LayerNorm(d_model)
        self.layers = layers

    def forward(self, x, enc_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.lnorm(x)


class PredictionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.pred = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.softmax(self.pred(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        src_embeds: Embeddings,
        tgt_embeds: Embeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        encoder: Encoder,
        decoder: Decoder,
        pred_layer: PredictionLayer,
    ):
        super().__init__()
        self.src_embeds = src_embeds
        self.tgt_embeds = tgt_embeds
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.pred_layer = pred_layer

    def forward(self, src, src_mask, tgt, tgt_mask):
        src = self.src_embeds(src)
        src = self.src_pos(src)
        enc_out = self.encoder(src, src_mask)

        tgt = self.tgt_embeds(tgt)
        tgt = self.tgt_pos(tgt)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        return self.pred_layer(dec_out)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int,
    N: int,
    h: int,
    dropout: float,
    d_ff: int,
):
    src_embed = Embeddings(vocab_size=src_vocab_size, d_model=d_model)
    tgt_embed = Embeddings(vocab_size=tgt_vocab_size, d_model=d_model)

    src_pos = PositionalEncoding(seq_len=src_seq_len, d_model=d_model, dropout=dropout)
    tgt_pos = PositionalEncoding(seq_len=tgt_seq_len, d_model=d_model, dropout=dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        encoder_block = EncoderBlock(
            self_attention_block, feed_forward_block, d_model, dropout
        )
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model, h, dropout)
        cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            self_attention_block,
            cross_attention_block,
            feed_forward_block,
            d_model,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks), d_model)
    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model)

    pred_layer = PredictionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        src_embed, tgt_embed, src_pos, tgt_pos, encoder, decoder, pred_layer
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
