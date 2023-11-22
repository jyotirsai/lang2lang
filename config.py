def get_config():
    return {
        "datasource": "iwslt2017",
        "input_lang": "en",
        "output_lang": "fr",
        "tokenizer_file": "tokenizer_{0}.json",
        "seq_len": 150,
        "batch_size": 256,
        "embed_size": 512,
        "dropout": 0.1,
        "num_heads": 8,
        "hidden_size": 2048,
        "num_layers": 6,
    }
