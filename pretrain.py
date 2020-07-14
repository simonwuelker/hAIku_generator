#https://medium.com/swlh/introduction-to-lstms-and-neural-network-text-generation-bd47adaf55fe
#hAIku poem generator
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.utils.data	#cant inherit from torch.utils.data.Dataset otherwise
import numpy as np
import random

import Generator
from Dataset import Dataset

import matplotlib.pyplot as plt
import string
import warnings

softmax = nn.Softmax(dim=0)

def generateHaiku(seed, num_haikus, length):
	"""Generates a certain number of haikus with a given length starting from some specified words"""
	for haiku_ix in range(num_haikus):
		text = seed
    	#generate the missing words to complete the haiku
		for i in range(length):
			generator.reset_hidden(batch_size = 1)	#every step is essentially a new forward pass

			input = torch.tensor([dataset.word_to_ix[word] for word in text.split()])
			outputs = generator(input.view(-1, 1, 1))
			index = sample_from_output(outputs[-1])
			text = f"{text} {dataset.ix_to_word[index.item()]}"

		print(f"Haiku Nr.{haiku_ix}:{text}")

def sample_from_output(prob):
	"""samples one element from a given log probability distribution"""
	#top k oder nucleus ist auch gut, einfach mal durchprobieren!
	index = torch.multinomial(torch.exp(prob), num_samples=1)
	return index


modelsave_path = "models/"
batch_size = 1
assert batch_size == 1	#padding not implemented
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)

#Init model
generator = Generator.generator(in_size = len(dataset.unique_tokens), out_size = len(dataset.unique_tokens))
try:
	generator.load_state_dict(torch.load("models/Generator_pretrained.pt"))
except:
	warnings.warn("Failed to load Generator Model")

#TRAINING
generator.train()
try:
	for epoch in range(1):
		print(f"Epoch: {epoch}")
		count = 0
		for input, target in dataloader:
			print(count)
			count += 1
			if count % 50 == 0:
				generateHaiku("i", num_haikus = 2, length = 7)
				
			generator.reset_hidden(batch_size)
			output = generator(input[0].long())
			target = target.squeeze()
			loss = generator.criterion(output, target.long())

			generator.optimizer.zero_grad()
			loss.backward()
			generator.optimizer.step()

			generator.losses.append(loss.item())
	
finally:	
	#Models are always saved, even after a KeyboardInterrupt
	torch.save(generator.state_dict(), f"{modelsave_path}Generator_pretrained.pt")

	#TESTING
	generator.eval()
	with torch.no_grad():
		generateHaiku("memorial", num_haikus = 10, length = 7)

	#plot the graph of the different losses over time
	fig, ax = plt.subplots()
	ax.plot(generator.losses, label = "Generator")
	plt.ylabel("Loss")
	plt.xlabel("training duration")
	ax.legend()

	plt.show()