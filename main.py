# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Generator
import Discriminator
import Tools
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import trange

# using a set seed for Reproducibility
torch.manual_seed(1)
np.random.seed(1)

modelsave_path = "models/"
batch_size = 1

dataset = Dataset(path_data="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init models
generator = Generator.generator(lr_actor=0.01, lr_critic=0.01, n_actions=dataset.embedding_dim)
discriminator = Discriminator.discriminator(in_size=dataset.embedding_dim)

# load models
# generator.loadModels()
discriminator.loadModel()

# TRAINING
generator.train()
discriminator.train()

try:
	for epoch in trange(10):
		for real_sample in dataloader:
			fake_sample = generator.generate(dataset, batch_size)
			print(dataset.decode(real_sample))
			print(dataset.decode(fake_sample))

			# take outputs from discriminator
			score_real = discriminator(real_sample)
			score_fake = discriminator(fake_sample.detach())

			# Save scores for evaluation
			discriminator.scores_real.append(score_real.item())
			discriminator.scores_fake.append(score_fake.item())

			# calculate loss
			loss_d = torch.mean(-torch.log(1 - score_fake) - torch.log(score_real))
			discriminator.losses.append(loss_d.item())

			# optimize discriminator
			discriminator.optimizer.zero_grad()
			loss_d.backward()
			discriminator.optimizer.step()

finally:
	# Models are always saved, even after a KeyboardInterrupt
	generator.saveModels()
	discriminator.saveModel()

	# TESTING
	discriminator.eval()
	generator.eval()

	with torch.no_grad():
		haikus = dataset.decode(generator.generate(dataset, 10))
		for haiku in haikus:
			print(haiku)

	# plot the graph of the different losses over time
	fig, ax = plt.subplots()
	ax.plot(discriminator.scores_real, label="Real")
	ax.plot(discriminator.scores_fake, label="Fake")
	# ax.plot(generator.losses[2:], label="Generator Loss")
	plt.ylabel("Loss")
	plt.xlabel("Episodes")
	ax.legend()

	plt.show()
