import torch
import torch.nn as nn
from pathlib import Path
from dataset import build_dataloader_and_tokenizers
from model import build_transformer
from utils import load_config, weights_file_path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    (
        train_dataloader,
        val_dataloader,
        src_tokenizer,
        tgt_tokenizer,
    ) = build_dataloader_and_tokenizers(config)
    model = build_transformer(
        src_tokenizer.get_vocab_size(),
        tgt_tokenizer.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
        config["n_times"],
        config["h"],
        config["dropout"],
        config["d_ff"],
    ).to(device)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = weights_file_path(config, config["preload"])
        print(f"Preloading model: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=src_tokenizer.token_to_id("<PAD>"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch: {epoch}"):
            enc_input = batch["encoder_input"].to(device)
            dec_input = batch["decoder_input"].to(device)
            enc_mask = batch["encoder_mask"].to(device)
            dec_mask = batch["decoder_mask"].to(device)

            output = model.forward(enc_input, enc_mask, dec_input, dec_mask)

            label = batch["label"].to(device)

            loss = loss_fn(
                output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1)
            )
            print(f"loss: {loss.item()}")

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_filename = weights_file_path(config, f"{epoch}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    config = load_config("./config.yaml")
    train_model(config)
