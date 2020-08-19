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

# generator.loadModel()
# discriminator.loadModel()

# TRAINING
generator.train()
discriminator.train()
try:
	for epoch in trange(1000):
		for real_sample in dataloader:
			fake_sample = generator.generate(batch_size=batch_size)
			print(dataset.decode(fake_sample))

			# take outputs from discriminator and log them
			score_real = discriminator(real_sample)
			score_fake = discriminator(fake_sample)

			# Save scores for evaluation
			discriminator.scores_real.append(score_real.item())
			discriminator.scores_fake.append(score_fake.item())

			# calculate loss
			loss_d = torch.mean(-torch.log(1 - score_fake) - torch.log(score_real))
			discriminator.losses.append(loss_d.item())

			# optimize models
			generator.learn(fake_sample, discriminator)
			discriminator.learn(loss_d)

finally:
	# Models are always saved, even after a KeyboardInterrupt
	generator.saveModel()
	discriminator.saveModel()

	# TESTING
	discriminator.eval()
	generator.eval()

	with torch.no_grad():
		pass

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
	fig.tight_layout()
	plt.savefig("training_graphs/main")
	plt.show()
