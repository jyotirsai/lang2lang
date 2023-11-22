from pathlib import Path


# code from https://github.com/hkproj/pytorch-transformer/blob/main/config.py
def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(config['model_folder'] + '/' + model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
