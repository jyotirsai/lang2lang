import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


class LanguageDataset(Dataset):
    def __init__(
        self, dataset, seq_len, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang
    ):
        super().__init__()
        self.dataset = dataset
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

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)


def retrieve_sentence(config, dataset):
    for sentence in dataset:
        yield sentence["translation"][config["lang"]]


def build_tokenizer(config, dataset):
    tokenizer_path = Path(config["tokenizer_file"].format(config["lang"]))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<UNK>", "<PAD>", "<SOS>", "<EOS>"])
        tokenizer.train_from_iterator(
            retrieve_sentence(dataset, config), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer
