# https://medium.com/swlh/introduction-to-lstms-and-neural-network-text-generation-bd47adaf55fe
# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Generator
from Dataset import Dataset
import Tools

import matplotlib.pyplot as plt
import warnings
from tqdm import trange


def generateHaiku(seed, num_haikus, length):
	"""Generates a certain number of haikus with a given length starting from some specified words"""
	seed = dataset.encode(seed)
	for haiku_ix in range(num_haikus):
		result = torch.zeros(1, length, 1)
		result[:, :seed.shape[1]] = seed

		# generate the missing words to complete the haiku
		for i in range(length - seed.shape[1]):
			generator.reset_hidden(batch_size=1)  # every step is essentially a new forward pass

			output = generator(result[:, :i + 1, :])[-1]
			index = Tools.sample_from_output(output)
			result[0, i + 1] = index

		print(f"Haiku Nr.{haiku_ix}:{dataset.decode(result)}")


batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init/Load model
generator = Generator.generator(alpha=0.01, beta=0.01, in_size=len(dataset.unique_tokens), embedding_dim=50)
generator.loadModels()

# TRAINING
generator.train()
try:
	for epoch in trange(0):
		total_loss = 0
		for sample in dataloader:
			generator.reset_hidden(batch_size)

			input = sample[:, :-1]
			target = sample[:, 1:]

			output = generator(input.long())
			target = target.squeeze()
			loss = generator.criterion(output, target.long())
			total_loss += loss.item()

			generator.optimizer.zero_grad()
			loss.backward()
			generator.optimizer.step()

		generator.losses.append(total_loss)

finally:
	# Models are always saved, even after a KeyboardInterrupt
	torch.save(generator.state_dict(), "models/Generator_pretrained.pt")

	# plot the graph of the different losses over time
	fig, ax = plt.subplots()
	ax.plot(generator.losses, label="Generator")
	plt.ylabel("Loss")
	plt.xlabel("Epochs")
	ax.legend()

	plt.show()

	# TESTING
	generator.eval()

	with torch.no_grad():
		generateHaiku("memorial", num_haikus=10, length=7)
