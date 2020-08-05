# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np
import random

import Generator
import Discriminator
import Tools
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
	for epoch in trange(20):
		for real_sample in dataloader:
			fake_sample = generator.generate(batch_size=batch_size)

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
	plt.plot(discriminator.scores_real, label = "Real")
	plt.plot(discriminator.scores_fake, label = "Fake")
	plt.ylabel("Scores")
	plt.xlabel("Training steps")
	plt.legend()
	plt.savefig("training_graphs/discriminator_scores_main")
	# plt.show()
	print(generator.losses)
