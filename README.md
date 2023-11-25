# lang2lang

Python library to easily build and train an encoder-decoder model based on the Transformer architecture outlined in "Attention is All You Need" by Vaswani et. al. Wrote the model from scratch using PyTorch and configured experiment tracking with MLflow. The model.py file defines each "block" of the network as a custom PyTorch Module subclass. Here's
the diagram from the original paper for reference.

<img src="transformer.png" alt="Transformer model" width="500" height="600">

The dataset is based on the iswlt2017 dataset and is applicable for all it's different language variations. To train
on other datasets, the dataset.py file will have to be configured for that particular dataset.

Work in progress.
