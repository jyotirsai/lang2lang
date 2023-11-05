import yaml
from pathlib import Path


def load_config(config_file):
    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def weights_file_path(config, epochs):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epochs}.pt"
    return str(Path(".") / model_folder / model_filename)
