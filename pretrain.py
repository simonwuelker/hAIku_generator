# https://medium.com/swlh/introduction-to-lstms-and-neural-network-text-generation-bd47adaf55fe
# hAIku poem generator
import torch
import torch.nn as nn
import torch.utils.data	 # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Generator
from Dataset import Dataset
import Tools
import torch.nn.functional as F

import matplotlib.pyplot as plt
import warnings

softmax = nn.Softmax(dim=0)


def generateHaiku(seed, num_haikus, length):
	"""Generates a certain number of haikus with a given length starting from some specified words"""
	for haiku_ix in range(num_haikus):
		text = seed
		# generate the missing words to complete the haiku
		for i in range(length - len(seed)):
			generator.reset_hidden(batch_size=1)  # every step is essentially a new forward pass

			input = F.one_hot(torch.tensor([dataset.token_to_ix[token] for token in text]), len(dataset.unique_tokens))
			outputs = generator(input.view(1, -1, len(dataset.unique_tokens)).float())
			index = Tools.sample_from_output(outputs[-1])
			text = f"{text}{dataset.ix_to_token[index.item()]}"

		print(f"Haiku Nr.{haiku_ix}:{text}.")


modelsave_path = "models/"
batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init model
generator = Generator.generator(in_size=len(dataset.unique_tokens), hidden_size = 600, n_layers=3, out_size=len(dataset.unique_tokens), batch_first=True)
try:
	generator.load_state_dict(torch.load("models/Generator_pretrained.pt"))
except FileNotFoundError:
	warnings.warn("Generator model does not exist")
except RuntimeError:
	warnings.warn("Failed to load Generator model")

# TRAINING
generator.train()
try:
	for epoch in range(5000):
		print(f"Epoch: {epoch}")
		count = 0
		for sample, target in dataloader:
			count += 1

			if count % 50 == 0:
				generateHaiku("i", num_haikus=4, length=20)

			input = sample[:, :-1, :]
			target = torch.argmax(sample[:, 1:, :].squeeze(), dim=1)

			generator.reset_hidden(batch_size)
			output = generator(input)  # for some reason the result does not have a batch dim

			loss = generator.criterion(output, target.long())

			# optimize generator
			generator.optimizer.zero_grad()
			loss.backward()
			generator.optimizer.step()

			generator.losses.append(loss.item())

finally:
	# Models are always saved, even after a KeyboardInterrupt
	torch.save(generator.state_dict(), f"{modelsave_path}Generator_pretrained.pt")

	# TESTING
	generator.eval()
	with torch.no_grad():
		generateHaiku("i", num_haikus=10, length=20)

	# plot the graph of loss over time
	fig, ax = plt.subplots()
	ax.plot(generator.losses, label="Generator")
	plt.ylabel("Loss")
	plt.xlabel("training duration")
	ax.legend()

	plt.show()
	print(generator.losses)
