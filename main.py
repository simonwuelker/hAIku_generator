import torch
import numpy as np

from Generator import Generator
from Discriminator import Discriminator
from Dataset import Dataset
from pretraining import discriminator_pretrain, generator_pretrain, word2vec_pretrain

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Define an ArgumentParser
parser = argparse.ArgumentParser(description="Train a SeqGAN model in continuous Action Space")

# training arguments (batch size, epochs)
parser.add_argument("--pretrain_w2v", help="Whether to pretrain the Word2Vec Models", action="store_true")
parser.add_argument("--pretrain_gen", help="Whether to pretrain the generator. Values are [batch size, epochs]", default=None, nargs=2)
parser.add_argument("--pretrain_dis", help="Whether to pretrain the discriminator. Values are [batch size, epochs]", default=None, nargs=2, type=int)
parser.add_argument("--no_train", help="Whether to train the models in GAN fashion.", action="store_true")

# Paths
parser.add_argument("--data", help="Specify path to the Dataset", default="data/dataset_clean.txt", dest="data_path")
parser.add_argument("--models", help="Specify path to the model Directory", default="models", dest="model_path")
parser.add_argument("--img", help="Specify path to store the training images in", default="training_img", dest="img_path")
parser.add_argument("--use_pretrained", help="Whether to use the pretrained Models", action="store_true", dest="use_pretrained")
parser.add_argument("--use_trained", help="Whether to use the trained Models", action="store_true", dest="use_trained")

# Parameters
parser.add_argument("--seed", help="Seed for torch/numpy", default=1, dest="seed")
parser.add_argument("--embedding_dim", help="Number of dimensions for word embeddings", type=int, default=80)
parser.add_argument("--epochs", help="Number of epochs during main training", type=int, default=1)
parser.add_argument("--batch_size", help="Batch size during main training", type=int, default=5)

# parse the provided arguments
args = parser.parse_args()

# set Seed
torch.manual_seed(int(args.seed))
np.random.seed(int(args.seed))

# initialize models
generator = Generator(args.embedding_dim, args.model_path)
discriminator = Discriminator(args.embedding_dim, args.model_path)

# load the models
if args.use_trained:
	generator.loadModel(path=generator.trained_path)
	discriminator.loadModel(path=discriminator.trained_path)

elif args.use_pretrained:
	generator.loadModel()
	discriminator.loadModel()

# optionally pretrain the Word2Vec model
if args.pretrain_w2v:
	word2vec_pretrain.train(args)

# create Dataset
dataset = Dataset(args)

# Generator Pretraining
if args.pretrain_gen is not None:
	generator = generator_pretrain.train(generator, dataset, args)
	generator.saveModel(generator.pretrained_path)

# Discriminator pretraining
if args.pretrain_dis is not None:
	discriminator = discriminator_pretrain.train(discriminator, dataset, args)
	discriminator.saveModel(discriminator.pretrained_path)

# Main Training
if not args.no_train:
	# TRAINING
	generator.train()
	discriminator.train()
	training_progress = tqdm(total = len(dataset) * args.epochs, desc = "Training")
	training_iterator = dataset.DataLoader(len(dataset), batch_size=args.batch_size)

	try:
		for epoch in range(int(args.epochs)):
			for real_sample in training_iterator:
				fake_sample = generator.generate(args.batch_size)
				print(dataset.decode(fake_sample))
				print(dataset.decode(real_sample))
				input()

				# update the progress bar
				training_progress.update(args.batch_size)

				# take outputs from discriminator
				score_real = discriminator(real_sample)
				score_fake = discriminator(fake_sample)

				# Save scores for evaluation
				discriminator.scores_real.append(score_real.mean().item())
				discriminator.scores_fake.append(score_fake.mean().item())

				# calculate loss
				loss_d = torch.mean(-torch.log(1.0001 - score_fake) - torch.log(0.0001 + score_real))
				discriminator.losses.append(loss_d.item())

				# optimize Models
				discriminator.learn(loss_d)
				generator.learn(fake_sample, discriminator)

	finally:
		# save the models
		generator.saveModel()
		discriminator.saveModel()

		# # TESTING
		# discriminator.eval()
		# generator.eval()

		# with torch.no_grad():
		# 	haikus = dataset.decode(generator.generate(10))
		# 	for haiku in haikus:
		# 		print(haiku)

		# smooth out the loss functions (avg of last 25 episodes)
		generator.losses = [np.mean(generator.losses[max(0, t-25):(t+1)]) for t in range(len(generator.losses))]
		discriminator.losses = [np.mean(discriminator.losses[max(0, t-25):(t+1)]) for t in range(len(discriminator.losses))]
		discriminator.scores_real = [np.mean(discriminator.scores_real[max(0, t-25):(t+1)]) for t in range(len(discriminator.scores_real))]
		discriminator.scores_fake = [np.mean(discriminator.scores_fake[max(0, t-25):(t+1)]) for t in range(len(discriminator.scores_fake))]


		# plot the graph of the different losses over time
		fig, ((d_score_plot, g_loss_plot), (d_loss_plot, _)) = plt.subplots(2, 2, num = "Training Data")

		# Discriminator scores
		d_score_plot.title.set_text("Discriminator Scores")
		d_score_plot.plot(discriminator.scores_real, label = "Real")
		d_score_plot.plot(discriminator.scores_fake, label = "Fake")
		d_score_plot.legend()

		# Generator Loss
		g_loss_plot.title.set_text("Generator Loss")
		g_loss_plot.plot(generator.losses, label = "Generator Loss")

		# Discriminator Loss
		d_loss_plot.title.set_text("Discriminator Loss")
		d_loss_plot.plot(discriminator.losses, label = "Discriminator Loss")

		fig.tight_layout()
		plt.savefig(f"{args.img_path}/main.png")
		plt.show()