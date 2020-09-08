# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Generator
import Discriminator
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import trange

# using a set seed for Reproducibility
torch.manual_seed(1)
np.random.seed(1)

batch_size = 1
modelsave_path = "models/"

dataset = Dataset(path_data="data/small_dataset.txt", train_test=1)
training_iterator = dataset.DataLoader(end=dataset.train_cap, batch_size=batch_size)

# Init models
generator = Generator.generator(dataset.embedding_dim)
discriminator = Discriminator.discriminator(in_size=dataset.embedding_dim)
generator.loadModel()
discriminator.loadModel()

# TRAINING
generator.train()
discriminator.train()
epochs = 1
training_progress = tqdm(total = dataset.train_cap * epochs, desc = "Training")

try:
	for epoch in range(epochs):
		for real_sample in training_iterator:
			fake_sample = generator.generate(batch_size)

			# update the progress bar
			training_progress.update(batch_size)

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

	# TESTING
	discriminator.eval()
	generator.eval()

	with torch.no_grad():
		haikus = dataset.decode(generator.generate(10))
		for haiku in haikus:
			print(haiku)

	# smooth out the loss functions (avg of last 25 episodes)
	generator.losses = [np.mean(generator.losses[max(0, t-25):(t+1)]) for t in range(len(generator.losses))]
	discriminator.losses = [np.mean(discriminator.losses[max(0, t-25):(t+1)]) for t in range(len(discriminator.losses))]

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
