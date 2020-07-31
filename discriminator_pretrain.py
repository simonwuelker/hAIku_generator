# https://medium.com/swlh/introduction-to-lstms-and-neural-network-text-generation-bd47adaf55fe
# hAIku poem generator
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

dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init/Load model
discriminator = Discriminator.discriminator(in_size=len(dataset.unique_tokens))
# discriminator.loadModel()

# TRAINING
discriminator.train()
try:
	for epoch in trange(50):
		total_loss = 0
		total_score_real = 0
		total_score_fake = 0
		for real_sample in dataloader:
			fake_sample = torch.tensor(np.random.randint(len(dataset.unique_tokens), size=real_sample.shape))

			# REAL SAMPLE
			discriminator.reset_hidden(batch_size)

			output = discriminator(real_sample)
			target = torch.ones(output.shape)
			loss = discriminator.criterion(output, target)
			total_score_real += output.item()
			

			# FAKE SAMPLE
			discriminator.reset_hidden(batch_size)

			output = discriminator(fake_sample.long())
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
	discriminator.saveModel()

	# plot the graph of the different losses over time
	# fig, ax = plt.subplots()
	# ax.plot(discriminator.losses, label="Discriminator Loss")
	plt.plot(discriminator.scores_real, label="Real")
	plt.plot(discriminator.scores_fake, label="Fake")
	plt.ylabel("Avg. Score")
	plt.xlabel("Epoch")
	plt.legend()
	plt.savefig("training_graphs/disc_pretrain_scores")
	plt.show()

	# TESTING
	discriminator.eval()

	with torch.no_grad():
		pass
		
