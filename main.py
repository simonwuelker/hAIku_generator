# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Generator
import Discriminator
from Dataset import Dataset
from pretraining import discriminator_pretrain, generator_pretrain, word2vec_pretrain
import gan_training

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Define an argument parser
parser = argparse.ArgumentParser(description="Train a SeqGAN model in continuous Action Space")

# training arguments (batch size, epochs)
parser.add_argument("--pretrain_w2v", help="Whether to pretrain the Word2Vec Models", action="store_true")
parser.add_argument("--pretrain_gen", help="Whether to pretrain the generator. Values are [batch size, epochs]", default=None, nargs=2)
parser.add_argument("--pretrain_dis", help="Whether to pretrain the discriminator. Values are [batch size, epochs]", default=None, nargs=2, type=int)
parser.add_argument("--train", help="Whether to train the models in GAN fashion. Values are [batch size, epochs]", default=None, nargs=2)
# Paths
parser.add_argument("--data", help="Specify path to the Dataset", default="data/dataset_clean.txt", dest="data_path")
parser.add_argument("--models", help="Specify path to the model Directory", default="models", dest="model_path")
parser.add_argument("--img", help="Specify path to store the training images in", default="training_img", dest="img_path")
parser.add_argument("--use_pretrained", help="Whether to use the pretrained Models", action="store_true", dest="use_pretrained")
parser.add_argument("--use_trained", help="Whether to use the trained Models", action="store_true", dest="use_trained")
# Parameters(add more here)
parser.add_argument("--seed", help="Seed for torch/numpy", default=1, dest="seed", type=int)
parser.add_argument("--embedding_dim", help="Number of dimensions for word embeddings", type=int, default=80)

# parse the provided arguments
args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# initialize models
generator = Generator.generator(args.embedding_dim, args.model_path)
discriminator = Discriminator.discriminator(args.embedding_dim, args.model_path)

# load the models
if args.use_trained:
	generator.loadModel(path=generator.trained_path)
	discriminator.loadModel(path=discriminator.trained_path)

elif args.use_pretrained:
	generator.loadModel()
	discriminator.loadModel()

# optionally pretrain the word2vec model
if args.pretrain_w2v:
	embedding = word2vec_pretrain.train(args)
else:
	embedding = torch.load(f"{args.model_path}/word2vec.model")

# create the Dataset for future training
dataset = Dataset(args.data_path, embedding)

if args.pretrain_gen is not None:
	generator = generator_pretrain.train(generator, dataset, args)

if args.pretrain_dis is not None:
	discriminator = discriminator_pretrain.train(discriminator, dataset, args)

if args.train is not None:
	generator, discriminator = gan_training.train()

# save the models
generator.saveModel()
discriminator.saveModel()