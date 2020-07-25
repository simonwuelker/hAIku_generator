# https://medium.com/swlh/introduction-to-lstms-and-neural-network-text-generation-bd47adaf55fe
# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Discriminator
from Dataset import Dataset

import matplotlib.pyplot as plt
import warnings
from tqdm import trange


batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init/Load model
discriminator = Discriminator.discriminator(in_size=len(dataset.unique_tokens))
try:
	discriminator.load_state_dict(torch.load("models/Discriminator_pretrained.pt"))
except FileNotFoundError:
	warnings.warn("Discriminator model does not exist")
except RuntimeError:
	warnings.warn("Failed to load Discriminator model")

# TRAINING
discriminator.train()
try:
	for epoch in trange(50):
		total_loss = 0
		for real_sample in dataloader:
			fake_sample = torch.tensor(np.random.randint(len(dataset.unique_tokens), size=real_sample.shape))

			# REAL SAMPLE
			discriminator.reset_hidden(batch_size)

			output = discriminator(real_sample.long())
			target = torch.ones(output.shape)
			loss = discriminator.criterion(output, target)

			# FAKE SAMPLE
			discriminator.reset_hidden(batch_size)

			output = discriminator(fake_sample.long())
			target = torch.zeros(output.shape)
			loss += discriminator.criterion(output, target)

			# OPTIMIZING
			total_loss += loss.item()
			discriminator.optimizer.zero_grad()
			loss.backward()
			discriminator.optimizer.step()

		discriminator.losses.append(total_loss)

finally:
	# Models are always saved, even after a KeyboardInterrupt
	torch.save(discriminator.state_dict(), "models/Discriminator_pretrained.pt")

	# plot the graph of the different losses over time
	fig, ax = plt.subplots()
	ax.plot(discriminator.losses, label="Discriminator")
	plt.ylabel("Loss")
	plt.xlabel("Epochs")
	ax.legend()

	plt.show()

	# TESTING
	discriminator.eval()

	with torch.no_grad():
		pass
