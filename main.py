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


def example(haiku_length=8):
	"""returns one generated sequence and optimizes the generator"""
	generator.reset_hidden(batch_size=1)

	result = torch.randint(len(dataset.unique_tokens), size=(1, haiku_length, 1))

	# generate the missing words to complete the haiku
	for i in range(haiku_length - 1):
		generator.reset_hidden(batch_size=1)

		output = generator(result[:, :i + 1, :])[-1]
		# target = Tools.Q(input, output, haiku_length, discriminator, dataset)

		index = Tools.sample_from_output(output)
		result[0, i + 1] = index

	return result


modelsave_path = "models/"
batch_size = 1

torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init models
generator = Generator.generator(in_size=dataset.embedding_dim, out_size=dataset.embedding_dim)
discriminator = Discriminator.discriminator(in_size=dataset.embedding_dim)

# load models
generator.loadModel()
discriminator.loadModel()

# TRAINING
# generator.train()
discriminator.train()

try:
	for epoch in trange(10):
		for real_sample in dataloader:
			print(dataset.decode(real_sample))
			fake_sample = example()

			# take outputs from discriminator
			score_real = discriminator(real_sample)
			assert False
			score_fake = discriminator(fake_sample)

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
	torch.save(discriminator.state_dict(), "models/Discriminator.pt")
	torch.save(generator.state_dict(), "models/Generator.pt")

	# TESTING
	discriminator.eval()
	generator.eval()

	with torch.no_grad():
		# could probably just implement a batch system in example and avoid for loop
		for haiku_ix in range(10):
			print(f"Haiku Nr.{haiku_ix}: {dataset.decode(example())}")

	# plot the graph of the different losses over time
	fig, ax = plt.subplots()
	ax.plot(discriminator.scores_real, label="Real")
	ax.plot(discriminator.scores_fake, label="Fake")
	# ax.plot(generator.losses[2:], label="Generator Loss")
	plt.ylabel("Loss")
	plt.xlabel("training duration")
	ax.legend()

	plt.show()
