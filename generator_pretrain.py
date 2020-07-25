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
	for haiku_ix in range(num_haikus):
		text = seed
		# generate the missing words to complete the haiku
		for i in range(length - len(seed.split())):
			generator.reset_hidden(batch_size=1)  # every step is essentially a new forward pass

			input = torch.tensor([dataset.word_to_ix[word] for word in text.split()])
			outputs = generator(input.view(1, -1, 1))

			index = Tools.sample_from_output(outputs[-1])
			text = f"{text} {dataset.ix_to_word[index.item()]}"

		print(f"Haiku Nr.{haiku_ix}:{text}")


batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init/Load model
generator = Generator.generator(in_size=len(dataset.unique_tokens), out_size=len(dataset.unique_tokens))
try:
	generator.load_state_dict(torch.load("models/Generator_pretrained.pt"))
except FileNotFoundError:
	warnings.warn("Generator model does not exist")
except RuntimeError:
	warnings.warn("Failed to load Generator model")

# TRAINING
generator.train()
try:
	for epoch in trange(50):
		total_loss = 0
		for sample in dataloader:
			generator.reset_hidden(batch_size)

			input = sample[:, :-1, :]
			target = sample[:, 1:, :]

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
