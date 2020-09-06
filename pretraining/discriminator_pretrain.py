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
import random

def generate_random():
	fake_sample = " ".join([random.choice(tuple(dataset_train.model.wv.wv.wv.wv.vocab.keys())) for word in range(random.randint(8, 13))])
	fake_sample = dataset_train.encode(fake_sample)
	fake_sample = fake_sample.view(1, -1, dataset_train.embedding_dim)
	return fake_sample

batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset_train = Dataset(path_data="../data/dataset_clean.txt", path_model="../models/word2vec.model", length = 1000, offset = 0)
dataset_test = Dataset(path_data="../data/dataset_clean.txt", path_model="../models/word2vec.model", length = 100, offset = 1000)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

# Init/Load model
discriminator = Discriminator.discriminator(in_size=dataset_train.embedding_dim)
# discriminator.loadModel(path="../models/Discriminator_pretrained.pt")

# TRAINING
discriminator.train()
try:
	for epoch in trange(1):
		for real_sample in dataloader_train:
			fake_sample = generate_random()

			# Pass the samples through the discriminator
			score_real = discriminator(real_sample)
			score_fake = discriminator(fake_sample)

			# optimize
			loss = torch.mean(- torch.log(0.001 + score_real) - torch.log(1.001 - score_fake))
			discriminator.optimizer.zero_grad()
			loss.backward()
			discriminator.optimizer.step()

			# save results
			discriminator.scores_real.append(score_real.item())
			discriminator.scores_fake.append(score_fake.item())
			discriminator.losses.append(loss.item())

finally:
	print(discriminator.scores_fake)
	print(discriminator.scores_real)
	# Models are always saved, even after a KeyboardInterrupt
	discriminator.saveModel(path="../models/Discriminator.pt")

	# smooth out the loss functions (avg of last window episodes)
	window = 25
	discriminator.scores_real = [np.mean(discriminator.scores_real[max(0, t-window):(t+1)]) for t in range(len(discriminator.scores_real))]
	discriminator.scores_fake = [np.mean(discriminator.scores_fake[max(0, t-window):(t+1)]) for t in range(len(discriminator.scores_fake))]
	discriminator.losses = [np.mean(discriminator.losses[max(0, t-window):(t+1)]) for t in range(len(discriminator.losses))]

	# plot the results
	fig, (loss_plot, score_plot) = plt.subplots(2)
	loss_plot.plot(discriminator.losses)
	loss_plot.title.set_text("Loss")

	score_plot.plot(discriminator.scores_real, label="Real")
	score_plot.plot(discriminator.scores_fake, label="Fake")
	score_plot.legend()
	score_plot.title.set_text("Scores")

	fig.tight_layout()
	plt.savefig("../training_graphs/disc_pretrain_scores")
	plt.show()

	# TESTING
	discriminator.eval()

	with torch.no_grad():
		total_real_score = 0
		for real_sample in dataloader_test:
			total_real_score += discriminator(real_sample)
		mean_real_score = total_real_score / len(dataloader_test)

		print(f"The mean score for real samples from the training set is: {mean_real_score}")

		
