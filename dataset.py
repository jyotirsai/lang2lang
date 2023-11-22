import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset


class TranslationDataset(Dataset):
    def __init__(self, raw_dataset, input_lang, output_lang, input_tokenizer, output_tokenizer, seq_len):
        self.raw_dataset = raw_dataset
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([input_tokenizer.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.tensor([input_tokenizer.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token = torch.tensor([input_tokenizer.token_to_id("[PAD]")], dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        input_ids = self.input_tokenizer.encode(sample['translation'][self.input_lang]).ids
        output_ids = self.output_tokenizer.encode(sample['translation'][self.output_lang]).ids

        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]
        
        num_of_padding_tokens_input = self.seq_len - len(input_ids) - 2 # -2 for sos and eos
        num_of_padding_tokens_output = self.seq_len - len(output_ids) - 1 # -1 for sos
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_ids, dtype=torch.long),
                self.eos_token,
                torch.tensor([self.pad_token] * num_of_padding_tokens_input, dtype=torch.long),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(output_ids, dtype=torch.long),
                torch.tensor([self.pad_token] * num_of_padding_tokens_output, dtype=torch.long),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(output_ids, dtype=torch.long),
                self.eos_token,
                torch.tensor([self.pad_token] * num_of_padding_tokens_output, dtype=torch.long),
            ],
            dim=0,
        )

        padding_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        lookahead_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & self.causal_mask(decoder_input.size(0))

        return encoder_input, decoder_input, label, padding_mask, lookahead_mask


    def __len__(self):
        return len(self.raw_dataset)
    
    def causal_mask(self, size):
      mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
      return mask == 0


def yield_sample(dataset, lang):
    for sample in dataset:
        yield sample['translation'][lang]

def build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if Path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(yield_sample(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    return tokenizer


def build_dataloader_and_tokenizers(config):
    raw_dataset = load_dataset(config['datasource'], config['datasource']+'-'+config['input_lang']+'-'+config['output_lang'])

    input_tokenizer = build_tokenizer(config, raw_dataset['train'], config['input_lang'])
    output_tokenizer = build_tokenizer(config, raw_dataset['train'], config['output_lang'])

    train = TranslationDataset(raw_dataset['train'], config['input_lang'], config['output_lang'], input_tokenizer, output_tokenizer, config['seq_len'])
    validation = TranslationDataset(raw_dataset['validation'], config['input_lang'], config['output_lang'], input_tokenizer, output_tokenizer, config['seq_len'])
    
    train_dataloader = DataLoader(train, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(validation, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader, input_tokenizer, output_tokenizer
