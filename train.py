import torch
import torch.nn as nn
from model import Transformer
from pathlib import Path
from dataset import build_dataloader_and_tokenizers
from utils import get_weights_file_path, latest_weights_file_path
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from config import get_config


def train(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device: ", device)

    mlflow.start_run()

    train_dataloader, val_dataloader, input_tokenizer, output_tokenizer = build_dataloader_and_tokenizers(config)

    model = Transformer(input_tokenizer.get_vocab_size(), output_tokenizer.get_vocab_size(), config['embed_size'], config['seq_len'], config['num_heads'], config['hidden_size'], config['dropout'], config['num_layers'])
    model.to(device)

    # apply xavier initialization
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=input_tokenizer.token_to_id('[PAD]')).to(device)

    initial_epoch = 0
    global_step = 0
    best_epoch = 0
    prev_val_loss = float('inf')
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    print(model_filename)
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        best_epoch = state['best_epoch']
        prev_val_loss = state['prev_val_loss']
    else:
        print('No model to preload, starting from scratch')
    
    mlflow.log_param("learning_rate", config['learning_rate'])
    mlflow.log_param("batch_size", config['batch_size'])
    mlflow.log_param("epochs", config['num_epochs'])
    mlflow.log_param("embed_size", config['embed_size'])
    mlflow.log_param("hidden_size", config['hidden_size'])
    mlflow.log_param("num_heads", config['num_heads'])
    mlflow.log_param("num_layers", config['num_layers'])
    mlflow.log_param("dropout", config['dropout'])
    mlflow.log_param("seq_len", config['seq_len'])
    mlflow.log_param("datasource", config['datasource'])
    
    mlflow.pytorch.log_model(model, "models")
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0
        idx = 0
        best_epoch = 0
        for encoder_input, decoder_input, label, padding_mask, lookahead_mask in tqdm(train_dataloader, desc=f"Processing Epoch {epoch} for training"):
            encoder_input = encoder_input.to(device) # (batch_size, seq_len)
            decoder_input = decoder_input.to(device) # (batch_size, seq_len)
            label = label.to(device) # (batch_size, seq_len)
            padding_mask = padding_mask.to(device) # (batch_size, 1, 1, seq_len)
            lookahead_mask = lookahead_mask.to(device) # (batch_size, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, padding_mask)
            decoder_output = model.decode(decoder_input, encoder_output, padding_mask, lookahead_mask)
            output = model.predict(decoder_output)

            loss = loss_fn(output.reshape(-1, output.shape[-1]), label.reshape(-1))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            idx += 1
            if idx % 100 == 0:
              print("Train loss: ", train_loss / idx)
              mlflow.log_metric("Train loss", train_loss / idx)
            
        
        print(f"Epoch Training Loss { train_loss / idx}")
        mlflow.log_metric("Epoch Training Loss", train_loss / idx)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            idx = 0
            for encoder_input, decoder_input, label, padding_mask, lookahead_mask in tqdm(val_dataloader, desc=f"Processing Epoch {epoch} for validation"):
                encoder_input = encoder_input.to(device) # (batch_size, seq_len)
                decoder_input = decoder_input.to(device) # (batch_size, seq_len)
                label = label.to(device) # (batch_size, seq_len)
                padding_mask = padding_mask.to(device) # (batch_size, 1, 1, seq_len)
                lookahead_mask = lookahead_mask.to(device) # (batch_size, 1, seq_len, seq_len)

                encoder_output = model.encode(encoder_input, padding_mask)
                decoder_output = model.decode(decoder_input, encoder_output, padding_mask, lookahead_mask)
                output = model.predict(decoder_output)

                loss = loss_fn(output.reshape(-1, output.shape[-1]), label.reshape(-1))
                val_loss += loss.item()

                idx += 1
                if idx % 100 == 0:
                  print("Validation loss: ", val_loss / idx)
                  mlflow.log_metric("Validation loss", val_loss / idx)

        print(f"Epoch Validation Loss { val_loss / idx}")
        mlflow.log_metric("Epoch Validation Loss", val_loss / idx)

        if val_loss / len(val_dataloader) < prev_val_loss:
            best_epoch = epoch
            prev_val_loss = val_loss / len(val_dataloader)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'best_epoch': best_epoch,
            'prev_val_loss': prev_val_loss
        }, model_filename)
        mlflow.end_run()

if __name__ == "__main__":
    config = get_config()
    train(config)