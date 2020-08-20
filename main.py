# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np
import random

import Generator
import Discriminator
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import trange


modelsave_path = "models/"
batch_size = 1

torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init models
generator = Generator.generator(in_size=len(dataset.unique_tokens), out_size=len(dataset.unique_tokens))
discriminator = Discriminator.discriminator(in_size=len(dataset.unique_tokens))

# generator.loadModel(path=generator.chkpt_path)
# discriminator.loadModel(path=discriminator.chkpt_path)

# TRAINING
generator.train()
discriminator.train()
try:
	for epoch in trange(250):
		for real_sample in dataloader:
			fake_sample = generator.generate(batch_size=batch_size)

			# take outputs from discriminator and log them
			score_real = discriminator(real_sample)
			score_fake = discriminator(fake_sample)

			# Save scores for evaluation
			discriminator.scores_real.append(score_real.item())
			discriminator.scores_fake.append(score_fake.item())

			# calculate loss
			loss_d = torch.mean(-torch.log(1.0001 - score_fake) - torch.log(score_real))
			discriminator.losses.append(loss_d.item())

			# optimize models
			loss_g, action_mem, disc_reward, completed = generator.learn(fake_sample, discriminator)
			discriminator.learn(loss_d)
			#print(dataset.decode(fake_sample), score_fake.item(), loss_g.item(), action_mem.detach(), disc_reward, dataset.decode(completed))

finally:
	# Models are always saved, even after a KeyboardInterrupt
	generator.saveModel()
	discriminator.saveModel()

	# TESTING
	discriminator.eval()
	generator.eval()

	with torch.no_grad():
		haikus = dataset.decode(generator.generate(batch_size=10))
		for haiku_ix in range(10):
			print(f"Haiku Nr.{haiku_ix}: '{haikus[haiku_ix]}'")
		print(discriminator(dataset[0].view(1, -1, generator.out_size)))

	# smooth out the loss functions and discriminator scores(avg of last 25 episodes)
	generator.losses = [np.mean(generator.losses[max(0, t-25):(t+1)]) for t in range(len(generator.losses))]
	discriminator.losses = [np.mean(discriminator.losses[max(0, t-25):(t+1)]) for t in range(len(discriminator.losses))]
	discriminator.scores_real = [np.mean(discriminator.scores_real[max(0, t-25):(t+1)]) for t in range(len(discriminator.scores_real))]
	discriminator.scores_fake = [np.mean(discriminator.scores_fake[max(0, t-25):(t+1)]) for t in range(len(discriminator.scores_fake))]

	# plot the graph of the different losses over time
	fig, axs = plt.subplots(2, 2, num = "Training Data")

	# Discriminator scores
	axs[0, 0].title.set_text("Discriminator Scores")
	axs[0, 0].plot(discriminator.scores_real, label = "Real")
	axs[0, 0].plot(discriminator.scores_fake, label = "Fake")
	axs[0, 0].legend()

	# Generator Loss
	axs[0, 1].title.set_text("Generator Loss")
	axs[0, 1].plot(generator.losses, label = "Generator Loss")

	# Discriminator Loss
	axs[1, 1].title.set_text("Discriminator Loss")
	axs[1, 1].plot(discriminator.losses, label = "Discriminator Loss")

	# generator probability distribution
	axs[1, 0].title.set_text("1. Generator Prob. Distribution")
	out = generator(torch.zeros(1, 1, generator.out_size)).view(1, -1)
	probs = torch.true_divide(out, torch.sum(out, dim=1).view(1, 1))
	axs[1, 0].bar(np.arange(generator.out_size), height=probs.view(-1).detach().numpy())

	fig.tight_layout()
	plt.savefig("training_graphs/main")
	plt.show()
