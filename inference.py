import torch
import torch.nn as nn
from pathlib import Path
from config import get_config
from tokenizers import Tokenizer

def inference(raw_input):
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device: ", device)

    input_tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['input_lang']))))
    output_tokenizer = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['output_lang']))))

    model = Transformer(input_tokenizer.get_vocab_size(), output_tokenizer.get_vocab_size(), config['embed_size'], config['seq_len'], config['num_heads'], config['hidden_size'], config['dropout'], config['num_layers'])
    model.to(device)

    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    state = torch.load(model_filename, map_location=torch.device(device))
    model.load_state_dict(state['model_state_dict'])

    output = []
    model.eval()
    with torch.no_grad():
        input = input_tokenizer.encode("Hello")
        input = torch.cat([
            torch.tensor([input_tokenizer.token_to_id('[SOS]')], dtype=torch.long),
            torch.tensor(input.ids, dtype=torch.long),
            torch.tensor([input_tokenizer.token_to_id('[EOS]')], dtype=torch.long),
            torch.tensor([input_tokenizer.token_to_id('[PAD]')] * (config['seq_len'] - len(input.ids) - 2), dtype=torch.long)
        ], dim=0).to(device)
        padding_mask = (input != input_tokenizer.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(input, input_mask)

        decoder_input = torch.empty(1, 1).fill_(input_tokenizer.token_to_id('[SOS]')).type_as(input).to(device)

        while decoder_input.size(1) < config['seq_len']:
            lookahead_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(padding_mask).to(device)
            out = model.decode(decoder_input, encoder_output, padding_mask, lookahead_mask)

            prob = model.predict(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).long().to(device)], dim=1)

            output.append(output_tokenizer.decode([next_word.item()]))

            if next_word == output_tokenizer.token_to_id('[EOS]'):
                break
    
    return ' '.join(output)




