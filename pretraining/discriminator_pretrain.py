# enable imports from parent directory
import sys
import os
sys.path.append(os.path.realpath(".."))

import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Discriminator
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import trange

batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path_data="../data/small_dataset.txt", path_model="../models/word2vec.model", length = 3, offset = 3)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
for sample in dataloader:
	print(dataset.decode(sample))
assert False

# Init/Load model
discriminator = Discriminator.discriminator(in_size=len(dataset.unique_tokens))
discriminator.loadModel(path="../models/Discriminator_pretrained.pt")

# TRAINING
discriminator.train()
try:
	for epoch in trange(2):
		total_loss = 0
		total_score_real = 0
		total_score_fake = 0
		for real_sample in dataloader:
			fake_sample = torch.tensor(np.random.randint(len(dataset.unique_tokens), size=real_sample.shape), dtype=torch.float)

			# REAL SAMPLE
			discriminator.reset_hidden(batch_size)

			output = discriminator(real_sample)
			target = torch.ones(output.shape)
			loss = discriminator.criterion(output, target)
			total_score_real += output.item()
			

			# FAKE SAMPLE
			discriminator.reset_hidden(batch_size)

			output = discriminator(fake_sample)
			target = torch.zeros(output.shape)
			loss += discriminator.criterion(output, target)
			total_score_fake += output.item()
			

			# OPTIMIZING
			total_loss += loss.item()
			discriminator.optimizer.zero_grad()
			loss.backward()
			discriminator.optimizer.step()

		# save outputs
		discriminator.scores_real.append(total_score_real / len(dataloader))
		discriminator.scores_fake.append(total_score_fake / len(dataloader))
		discriminator.losses.append(total_loss / len(dataloader))

finally:
	# Models are always saved, even after a KeyboardInterrupt
	discriminator.saveModel(path="../models/Discriminator.pt")

	# plot the graph of the different losses over time
	# fig, ax = plt.subplots()
	# ax.plot(discriminator.losses, label="Discriminator Loss")
	plt.plot(discriminator.scores_real, label="Real")
	plt.plot(discriminator.scores_fake, label="Fake")
	plt.ylabel("Avg. Score")
	plt.xlabel("Epoch")
	plt.legend()
	plt.savefig("../training_graphs/disc_pretrain_scores")
	plt.show()

	# TESTING
	discriminator.eval()

	with torch.no_grad():
		pass
		
