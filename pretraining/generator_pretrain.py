# enable imports from parent directory
import sys
import os
sys.path.append(os.path.realpath(".."))

import torch
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import numpy as np

import Generator
from Dataset import Dataset
import Tools

import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm


batch_size = 1
torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path_data="../data/dataset_clean.txt", path_model="../models/word2vec.model", train_test=1)
dataloader = dataset.DataLoader(batch_size=batch_size, end=dataset.train_cap)

# Init/Load model
generator = Generator.generator(embedding_dim=dataset.embedding.embedding_dim)
# generator.loadModel()
std = torch.full([batch_size, dataset.embedding.embedding_dim], 1, dtype=torch.float32)  # manual std is used during training

# TRAINING
generator.train()
epochs = 1
training_progress = tqdm(total=dataset.train_cap * epochs, desc="Training")
try:
	for epoch in range(epochs):
		for packed_sample in dataloader:
			# update the progress bar
			training_progress.update(batch_size)

			# unpack the sample
			sample, lengths = pad_packed_sequence(packed_sample, batch_first=True)

			# calculate the loss based on 'how far off' the generator is on each token
			loss = 0
			for index in range(1, sample.shape[1] - 1):
				input = sample[:, :index]
				target = sample[: index]

				predicted_tokens = generator(input, std=std)

				# if the sequence has ended, dont count the loss
				for i, length in enumerate(lengths):
					if index <= length:
						loss += F.mse(predicted_tokens[i], target[i])

			# optimize the model
			generator.optimizer.zero_grad()
			loss.backward()
			generator.optimizer.step()
			generator.losses.append(loss.item())

finally:
	# Models are always saved, even after a KeyboardInterrupt
	generator.saveModel("../models/Generator_pretrained.pt")

	# plot generator loss
	plt.title("Loss")
	plt.plot(generator.losses)
	plt.ylabel("Loss")
	plt.xlabel("Epochs")
	plt.savefig("../training_graphs/generator_pretraining")
	plt.show()

	# TESTING
	generator.eval()

	with torch.no_grad():
		# generate 10 Haikus
		packed_haikus = generator.generate(batch_size=10, set_std=torch.full([10, dataset.embedding.embedding_dim], 1, dtype=torch.float32))

		# unpack sequence
		unpacked, lengths = pad_packed_sequence(packed_haikus)
		decoded = dataset.decode(unpacked)
		for haiku in decoded:
			print(haiku)

