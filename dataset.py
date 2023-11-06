import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset


class LanguageDataset(Dataset):
    def __init__(
        self, raw_dataset, seq_len, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang
    ):
        super().__init__()
        self.dataset = raw_dataset
        self.seq_len = seq_len
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # need <SOS>, <EOS>, <PAD> tokens to append to our data
        self.sos_token = torch.tensor(
            [src_tokenizer.token_to_id("<SOS>")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [src_tokenizer.token_to_id("<EOS>")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [src_tokenizer.token_to_id("<PAD>")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_text = self.dataset[index]["translation"][self.src_lang]
        tgt_text = self.dataset[index]["translation"][self.tgt_lang]

        enc_input_ids = self.src_tokenizer.encode(src_text).ids
        dec_input_ids = self.tgt_tokenizer.encode(tgt_text).ids

        num_enc_padding = self.seq_len - len(enc_input_ids) - 2  # <SOS>, <EOS>
        num_dec_padding = self.seq_len - len(dec_input_ids) - 1  # <SOS>

        if num_enc_padding < 0 or num_dec_padding < 0:
            raise ValueError("Sentence is too long")

        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_enc_padding, dtype=torch.int64),
            ]
        )

        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_ids, dtype=torch.int64),
                torch.tensor([self.pad_token] * num_dec_padding, dtype=torch.int64),
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_ids, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_dec_padding, dtype=torch.int64),
            ]
        )

        return {
            "encoder_input": enc_input,
            "decoder_input": dec_input,
            "encoder_mask": (enc_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, seq_len)
            "decoder_mask": (dec_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(dec_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


def retrieve_sentence(config, raw_dataset, lang):
    for sentence in raw_dataset:
        yield sentence["translation"][config[lang]]


def build_tokenizer(config, raw_dataset, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<UNK>", "<PAD>", "<SOS>", "<EOS>"])
        tokenizer.train_from_iterator(
            retrieve_sentence(raw_dataset, config, lang), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def build_dataloader_and_tokenizers(config):
    raw_dataset = load_dataset("opus_books", "en-fr", split="train")
    src_tokenizer = build_tokenizer(config, raw_dataset, config["lang_src"])
    tgt_tokenizer = build_tokenizer(config, raw_dataset, config["lang_tgt"])

    train_size = int(0.9 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    raw_train, raw_val = random_split(raw_dataset, [train_size, val_size])

    train = LanguageDataset(
        raw_train,
        config["seq_len"],
        src_tokenizer,
        tgt_tokenizer,
        config["src_lang"],
        config["tgt_lang"],
    )
    val = LanguageDataset(
        raw_val,
        config["seq_len"],
        src_tokenizer,
        tgt_tokenizer,
        config["src_lang"],
        config["tgt_lang"],
    )

    train_dataloader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer
