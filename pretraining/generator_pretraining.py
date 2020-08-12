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
# generator.loadModel()

# TRAINING
generator.train()
try:
	for epoch in trange(2):
		count = 0
		for real_sample in dataloader:
			total_loss = 0

			#let the generator predict every single character
			for index in range(real_sample.shape[1] - 1):
				if index == 0:
					input = torch.zeros(batch_size, 1, len(dataset.unique_tokens))
				else:
					input = real_sample[:, :index]

				target = real_sample[:, index]

				generator.reset_hidden(batch_size)
				output = generator(input)[:, -1]

				total_loss += generator.criterion(output, target)

			# optimize generator
			generator.optimizer.zero_grad()
			total_loss.backward()
			generator.optimizer.step()

			generator.losses.append(total_loss.item())

finally:
	# Models are always saved, even after a KeyboardInterrupt
	generator.saveModel(path="models/Generator_pretrained.pt")

	# TESTING
	generator.eval()
	with torch.no_grad():
		# add tests in here
		pass

	# plot the graph of loss over time
	plt.plot(generator.losses, label="Loss")
	plt.ylabel("Loss")
	plt.xlabel("Samples")
	plt.legend()

	plt.show()
	print(generator.losses)
