# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Generator
import Discriminator
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Define an argument parser
parser = argparse.ArgumentParser(description="Train a SeqGAN model in continuous Action Space")
parser.add_argument("--data", help="Specify path to the Dataset", default="data/dataset_clean.txt", dest="data_path")
parser.add_argument("--models", help="Specify path to the model Directory", default="models", dest="model_path")
parser.add_argument("--use_pretrained", help="Whether to use the pretrained Models", action="store_true", dest="use_pretrained")
parser.add_argument("--use_trained", help="Whether to use the trained Models", action="store_true", dest="use_trained")
parser.add_argument("--batch_size", help="Batch Size", dest="batch_size", default=1, type=int)
parser.add_argument("--seed", help="Seed for torch", default=1, dest="seed", type=int)
parser.add_argument("--epochs", help="Number of Training Epochs", default=1, dest="epochs", type=int)

# parse the provided arguments
args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)

dataset = Dataset(args.data_path, args.model_path, train_test=1)
training_iterator = dataset.DataLoader(end=dataset.train_cap, batch_size=args.batch_size)

# initialize models
generator = Generator.generator(dataset.embedding.embedding_dim, args.model_path)
discriminator = Discriminator.discriminator(dataset.embedding.embedding_dim, args.model_path)

# load the models
if args.use_trained:
	generator.loadModel(path=generator.chkpt_path)
	discriminator.loadModel(path=discriminator.chkpt_path)
elif args.use_pretrained:
	generator.loadModel()
	discriminator.loadModel()

# TRAINING
generator.train()
discriminator.train()
training_progress = tqdm(total = dataset.train_cap * args.epochs, desc = "Training")

try:
	for epoch in range(args.epochs):
		for real_sample in training_iterator:
			print(dataset.decode(real_sample))
			assert False
			fake_sample = generator.generate(args.batch_size)

			# update the progress bar
			training_progress.update(args.batch_size)

			# take outputs from discriminator
			score_real = discriminator(real_sample)
			score_fake = discriminator(fake_sample)

			# Save scores for evaluation
			discriminator.scores_real.append(score_real.item())
			discriminator.scores_fake.append(score_fake.item())

			# calculate loss
			loss_d = torch.mean(-torch.log(1 - score_fake) - torch.log(score_real))
			discriminator.losses.append(loss_d.item())

			# optimize discriminator
			discriminator.learn(loss_d)
			generator.learn(fake_sample, discriminator)

finally:
	# Models are always saved, even after a KeyboardInterrupt
	generator.saveModel()
	discriminator.saveModel()

	# # TESTING
	# discriminator.eval()
	# generator.eval()

	# with torch.no_grad():
	# 	haikus = dataset.decode(generator.generate(10))
	# 	for haiku in haikus:
	# 		print(haiku)

	# # smooth out the loss functions (avg of last 25 episodes)
	# generator.losses = [np.mean(generator.losses[max(0, t-25):(t+1)]) for t in range(len(generator.losses))]
	# discriminator.losses = [np.mean(discriminator.losses[max(0, t-25):(t+1)]) for t in range(len(discriminator.losses))]

	# # plot the graph of the different losses over time
	# fig, axs = plt.subplots(2, 2, num = "Training Data")

	# # Discriminator scores
	# axs[0, 0].title.set_text("Discriminator Scores")
	# axs[0, 0].plot(discriminator.scores_real, label = "Real")
	# axs[0, 0].plot(discriminator.scores_fake, label = "Fake")
	# axs[0, 0].legend()

	# # Generator Loss
	# axs[0, 1].title.set_text("Generator Loss")
	# axs[0, 1].plot(generator.losses, label = "Generator Loss")

	# # Discriminator Loss
	# axs[1, 1].title.set_text("Discriminator Loss")
	# axs[1, 1].plot(discriminator.losses, label = "Discriminator Loss")
	# fig.tight_layout()
	# plt.savefig("training_graphs/main")
	# plt.show()
