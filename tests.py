import torch
import torch.nn as nn
import unittest
from model import (
    Embeddings,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForwardBlock,
    EncoderBlock,
    Encoder,
    DecoderBlock,
    Decoder,
    PredictionLayer,
    Transformer,
    build_transformer,
)


class TestTransformer(unittest.TestCase):
    def test_embedding_shape(self):
        vocab_size = 10000
        d_model = 512
        max_seq_len = 20
        batch_size = 8  # Example batch size
        embeddings = Embeddings(vocab_size, d_model)

        input_data = torch.randint(0, vocab_size, (batch_size, max_seq_len))
        embedded = embeddings(input_data)
        expected_shape = (batch_size, max_seq_len, d_model)

        self.assertEqual(embedded.shape, expected_shape)

    def test_embedding_values(self):
        vocab_size = 10000
        d_model = 512
        batch_size = 8  # Example batch size
        embeddings = Embeddings(vocab_size, d_model)

        input_data = torch.randint(0, vocab_size, (batch_size, 20))
        embedded = embeddings(input_data)

        # Test if the embeddings are not all zeros
        self.assertNotAlmostEqual(torch.sum(embedded), 0, delta=1e-5)

    def test_positional_encoding_shape(self):
        seq_len = 20
        d_model = 512
        dropout = 0.1
        positional_encoder = PositionalEncoding(seq_len, d_model, dropout)

        input_data = torch.randn((1, seq_len, d_model))  # Example input sequence
        encoded = positional_encoder(input_data)
        expected_shape = (1, seq_len, d_model)

        self.assertEqual(encoded.shape, expected_shape)

    def test_multi_head_attention_shape(self):
        d_model = 512
        h = 8
        dropout = 0.1
        multi_head_attention = MultiHeadAttention(d_model, h, dropout)

        batch_size = 1
        seq_len = 20
        query = torch.randn((batch_size, seq_len, d_model))
        key = torch.randn((batch_size, seq_len, d_model))
        value = torch.randn((batch_size, seq_len, d_model))
        mask = None  # Example mask (if required)

        output = multi_head_attention(query, key, value, mask)
        expected_shape = (batch_size, seq_len, d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_feed_forward_block_shape(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        batch_size = 1
        seq_len = 20
        input_data = torch.randn((batch_size, seq_len, d_model))

        output = feed_forward_block(input_data)
        expected_shape = (batch_size, seq_len, d_model)

        self.assertEqual(output.shape, expected_shape)

    def test_feed_forward_block_dropout(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.5  # High dropout for testing
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        batch_size = 1
        seq_len = 20
        input_data = torch.randn((batch_size, seq_len, d_model))

        output = feed_forward_block(input_data)

        # Check if dropout is applied (values not equal to input)
        self.assertFalse(torch.allclose(output, input_data))

    def test_encoder_block_shape(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        encoder_block = EncoderBlock(
            MultiHeadAttention(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            d_model,
            dropout,
        )

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        src_mask = None  # Example source mask

        output = encoder_block(input_data, src_mask)
        expected_shape = (batch_size, seq_len, d_model)

        self.assertEqual(output.shape, expected_shape)

    def test_encoder_block_self_attention(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        encoder_block = EncoderBlock(
            MultiHeadAttention(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            d_model,
            dropout,
        )

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        src_mask = None  # Example source mask

        output = encoder_block(input_data, src_mask)

        # Ensure that self-attention is applied (values not equal to input)
        self.assertFalse(torch.allclose(output, input_data))

    def test_encoder_shape(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        num_layers = 6
        encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    MultiHeadAttention(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    d_model,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        encoder = Encoder(encoder_blocks, d_model)

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        mask = None  # Example mask

        output = encoder(input_data, mask)
        expected_shape = (batch_size, seq_len, d_model)

        self.assertEqual(output.shape, expected_shape)

    def test_encoder_forward(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        num_layers = 6
        encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    MultiHeadAttention(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    d_model,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        encoder = Encoder(encoder_blocks, d_model)

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        mask = None  # Example mask

        output = encoder(input_data, mask)

        # Ensure that the encoder applies all layers (values not equal to input)
        self.assertFalse(torch.allclose(output, input_data))

    def test_decoder_block_shape(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        decoder_block = DecoderBlock(
            MultiHeadAttention(d_model, h, dropout),
            MultiHeadAttention(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            d_model,
            dropout,
        )

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        enc_out = torch.randn((batch_size, seq_len, d_model))
        src_mask = None  # Example source mask
        tgt_mask = None  # Example target mask

        output = decoder_block(input_data, enc_out, src_mask, tgt_mask)
        expected_shape = (batch_size, seq_len, d_model)

        self.assertEqual(output.shape, expected_shape)

    def test_decoder_block_self_attention(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        decoder_block = DecoderBlock(
            MultiHeadAttention(d_model, h, dropout),
            MultiHeadAttention(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            d_model,
            dropout,
        )

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        enc_out = torch.randn((batch_size, seq_len, d_model))
        src_mask = None  # Example source mask
        tgt_mask = None  # Example target mask

        output = decoder_block(input_data, enc_out, src_mask, tgt_mask)

        # Ensure that self-attention is applied (values not equal to input)
        self.assertFalse(torch.allclose(output, input_data))

    def test_decoder_block_cross_attention(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        decoder_block = DecoderBlock(
            MultiHeadAttention(d_model, h, dropout),
            MultiHeadAttention(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            d_model,
            dropout,
        )

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        enc_out = torch.randn((batch_size, seq_len, d_model))
        src_mask = None  # Example source mask
        tgt_mask = None  # Example target mask

        output = decoder_block(input_data, enc_out, src_mask, tgt_mask)

        # Ensure that cross-attention is applied (values not equal to input)
        self.assertFalse(torch.allclose(output, input_data))

    def test_decoder_shape(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        num_layers = 6
        decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    MultiHeadAttention(d_model, h, dropout),
                    MultiHeadAttention(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    d_model,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        decoder = Decoder(decoder_blocks, d_model)

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        enc_out = torch.randn((batch_size, seq_len, d_model))
        src_mask = None  # Example source mask
        tgt_mask = None  # Example target mask

        output = decoder(input_data, enc_out, src_mask, tgt_mask)
        expected_shape = (batch_size, seq_len, d_model)

        self.assertEqual(output.shape, expected_shape)

    def test_decoder_forward(self):
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        num_layers = 6
        decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    MultiHeadAttention(d_model, h, dropout),
                    MultiHeadAttention(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    d_model,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        decoder = Decoder(decoder_blocks, d_model)

        batch_size = 1
        input_data = torch.randn((batch_size, seq_len, d_model))
        enc_out = torch.randn((batch_size, seq_len, d_model))
        src_mask = None  # Example source mask
        tgt_mask = None  # Example target mask

        output = decoder(input_data, enc_out, src_mask, tgt_mask)

        # Ensure that the decoder applies all layers (values not equal to input)
        self.assertFalse(torch.allclose(output, input_data))

    def test_prediction_layer_shape(self):
        d_model = 512
        vocab_size = 10000
        prediction_layer = PredictionLayer(d_model, vocab_size)

        batch_size = 1
        seq_len = 20
        input_data = torch.randn((batch_size, seq_len, d_model))

        output = prediction_layer(input_data)
        expected_shape = (batch_size, seq_len, vocab_size)

        self.assertEqual(output.shape, expected_shape)

    def test_prediction_layer_softmax(self):
        d_model = 512
        vocab_size = 10000
        prediction_layer = PredictionLayer(d_model, vocab_size)

        batch_size = 1
        seq_len = 20
        input_data = torch.randn((batch_size, seq_len, d_model))

        output = prediction_layer(input_data)

        # Check if the output is a valid probability distribution
        self.assertTrue(
            torch.allclose(
                torch.sum(output, dim=-1), torch.ones((batch_size, seq_len)), atol=1e-6
            )
        )

    def test_encoder_decoder_interaction(self):
        # Parameters
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        seq_len = 20
        num_layers = 6

        # Create Encoder Blocks
        encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    MultiHeadAttention(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    d_model,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Create Encoder and Decoder
        encoder = Encoder(encoder_blocks, d_model)
        decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    MultiHeadAttention(d_model, h, dropout),
                    MultiHeadAttention(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    d_model,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        decoder = Decoder(decoder_blocks, d_model)

        # Create fake input data
        batch_size = 1
        src_seq_len = 20
        tgt_seq_len = 30
        src_data = torch.randn((batch_size, src_seq_len, d_model))
        tgt_data = torch.randn((batch_size, tgt_seq_len, d_model))

        # Masking (replace with actual masks)
        src_mask = None
        tgt_mask = None

        # Forward pass
        encoder_output = encoder(src_data, src_mask)
        decoder_output = decoder(tgt_data, encoder_output, src_mask, tgt_mask)

        # Ensure that the interaction produces the expected shapes
        expected_encoder_shape = (batch_size, src_seq_len, d_model)
        expected_decoder_shape = (batch_size, tgt_seq_len, d_model)

        self.assertEqual(encoder_output.shape, expected_encoder_shape)
        self.assertEqual(decoder_output.shape, expected_decoder_shape)

    def test_transformer_shape(self):
        # Define model parameters
        src_vocab_size = 10000
        tgt_vocab_size = 10000
        d_model = 512
        d_ff = 2048
        dropout = 0.1
        h = 8
        src_seq_len = 20
        tgt_seq_len = 30
        num_layers = 6

        # Create the Embeddings and PositionalEncoding instances
        src_embeds = Embeddings(src_vocab_size, d_model)
        tgt_embeds = Embeddings(tgt_vocab_size, d_model)
        src_pos = PositionalEncoding(src_seq_len, d_model, dropout)
        tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)

        # Create the Encoder and Decoder instances
        encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    MultiHeadAttention(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    d_model,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        encoder = Encoder(encoder_blocks, d_model)

        decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    MultiHeadAttention(d_model, h, dropout),
                    MultiHeadAttention(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    d_model,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        decoder = Decoder(decoder_blocks, d_model)

        # Create the PredictionLayer instance
        pred_layer = PredictionLayer(d_model, tgt_vocab_size)

        # Create the Transformer model
        transformer = Transformer(
            src_embeds, tgt_embeds, src_pos, tgt_pos, encoder, decoder, pred_layer
        )

        # Create fake input data
        batch_size = 1
        src_data = torch.randint(
            0, src_vocab_size, (batch_size, src_seq_len)
        )  # Replace with actual data
        tgt_data = torch.randint(
            0, tgt_vocab_size, (batch_size, tgt_seq_len)
        )  # Replace with actual data

        # Masking (replace with actual masks)
        src_mask = None
        tgt_mask = None

        # Forward pass
        output = transformer(src_data, src_mask, tgt_data, tgt_mask)

        # Define expected output shape
        expected_shape = (batch_size, tgt_seq_len, tgt_vocab_size)

        self.assertEqual(output.shape, expected_shape)

    def test_build_transformer(self):
        # Hyperparameters
        src_vocab_size = 10000
        tgt_vocab_size = 10000
        src_seq_len = 20
        tgt_seq_len = 30
        d_model = 512
        N = 6
        h = 8
        dropout = 0.1
        d_ff = 2048

        # Build the Transformer model
        transformer = build_transformer(
            src_vocab_size,
            tgt_vocab_size,
            src_seq_len,
            tgt_seq_len,
            d_model,
            N,
            h,
            dropout,
            d_ff,
        )

        # Define fake input data
        batch_size = 1
        src_data = torch.randint(
            0, src_vocab_size, (batch_size, src_seq_len)
        )  # Replace with actual data
        tgt_data = torch.randint(
            0, tgt_vocab_size, (batch_size, tgt_seq_len)
        )  # Replace with actual data

        # Masking (replace with actual masks)
        src_mask = None
        tgt_mask = None

        # Forward pass
        output = transformer(src_data, src_mask, tgt_data, tgt_mask)

        # Ensure that the model is created and can make predictions
        self.assertIsNotNone(transformer)
        self.assertTrue(isinstance(transformer, Transformer))


if __name__ == "__main__":
    unittest.main()
