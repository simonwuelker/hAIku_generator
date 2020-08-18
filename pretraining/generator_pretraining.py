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
from tqdm import trange


batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init model
generator = Generator.generator(in_size=len(dataset.unique_tokens), out_size=len(dataset.unique_tokens))
generator.loadModel()

# TRAINING
generator.train()
try:
	for epoch in trange(1000):
		count = 0
		for real_sample in dataloader:
			loss = torch.zeros(real_sample.shape[1] - 1)

			#let the generator predict every single character
			for index in range(1, real_sample.shape[1]):
				generator.reset_hidden(batch_size)
				input = real_sample[:, :index]

				target = torch.argmax(real_sample[:, index], dim=1).view(batch_size)
				output = generator(input)[:, -1]

				loss[index - 1] = generator.criterion(output, target)

			mean_loss = torch.mean(loss)

			# optimize generator
			generator.optimizer.zero_grad()
			mean_loss.backward()
			generator.optimizer.step()

			generator.losses.append(mean_loss.item())

finally:
	# Models are always saved, even after a KeyboardInterrupt
	generator.saveModel(path="models/Generator_pretrained.pt")

	# TESTING
	generator.eval()
	with torch.no_grad():
		haikus = dataset.decode(generator.generate(batch_size=10, seed=7))
		for haiku in haikus:
			print(haiku)

	# plot the graph of loss over time
	plt.plot(generator.losses, label="Loss")
	plt.ylabel("Loss")
	plt.xlabel("Samples")
	plt.legend()

	plt.show()