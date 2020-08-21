# enable imports from parent directory
import sys
import os
sys.path.append(os.path.realpath(".."))

import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import torch.nn.functional as F
import numpy as np

import Discriminator
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import trange

batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="../data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init Model
discriminator = Discriminator.discriminator(in_size=len(dataset.unique_tokens))

# TRAINING
discriminator.train()
try:
	for epoch in trange(250):
		total_loss = 0
		total_score_real = 0
		total_score_fake = 0
		for real_sample in dataloader:
			fake_sample = F.one_hot(torch.randint(len(dataset.unique_tokens), size=[batch_size, real_sample.shape[1]]), len(dataset.unique_tokens)).float()

			# REAL SAMPLE
			score_real = discriminator(real_sample)
			total_score_real += score_real.item()

			# FAKE SAMPLE
			score_fake = discriminator(fake_sample)
			total_score_fake += score_fake.item()
			if score_fake.item() > 0.5:
				print(dataset.decode(fake_sample))
			
			# OPTIMIZING
			loss = torch.mean(-torch.log(1.001 - score_fake) - torch.log(score_real))
			total_loss += loss.item()
			discriminator.optimizer.zero_grad()
			loss.backward()
			discriminator.optimizer.step()

		# save mean outputs/loss
		discriminator.scores_real.append(total_score_real / len(dataloader))
		discriminator.scores_fake.append(total_score_fake / len(dataloader))
		discriminator.losses.append(total_loss / len(dataloader))
finally:
	# Models are always saved, even after a KeyboardInterrupt
	discriminator.saveModel(path="../models/Discriminator_pretrained.pt")

	# Graph the Loss as well as the scores in 2 subplots
	fig, axs = plt.subplots(2)

	# Discriminator Scores
	axs[0].title.set_text("Discriminator Scores")
	axs[0].plot(discriminator.scores_real, label="Real")
	axs[0].plot(discriminator.scores_fake, label="Fake")
	axs[0].legend()

	# Discriminator Loss
	axs[1].title.set_text("Discriminator Loss")
	axs[1].plot(discriminator.losses)

	plt.savefig("../training_graphs/discriminator_pretrain_scores")
	fig.tight_layout()
	plt.show()

	# TESTING
	discriminator.eval()

	with torch.no_grad():
		# add some tests in here
		pass
		
