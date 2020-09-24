import torch
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import numpy as np

import Generator
from Dataset import Dataset

import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm


def train(generator, dataset, args):
	batch_size = int(args.pretrain_gen[0])
	epochs = int(args.pretrain_gen[1])

	# define std during training
	std = torch.full([batch_size, dataset.embedding.embedding_dim], 1, dtype=torch.float32)  # manual std is used during training

	# TRAINING
	generator.train()
	training_iterator = dataset.DataLoader(len(dataset), batch_size=batch_size)
	training_progress = tqdm(total=len(dataset), desc="Training")
	try:
		for epoch in range(epochs):
			for packed_sample in training_iterator:
				# update the progress bar
				training_progress.update(batch_size)

				# unpack the sample
				sample, lengths = pad_packed_sequence(packed_sample, batch_first=True)

				# calculate the loss based on 'how far off' the generator is on each token
				loss = 0
				for index in range(1, sample.shape[1] - 1):
					input = sample[:, :index]
					target = sample[:, index]

					# get the distribution from the generator
					_, distribution = generator(input, lengths=lengths, std=std)
					print(distribution, distribution.sample().shape)
					assert False
					# if the sequence has ended, dont count the loss
					for batch_ix in range(batch_size):
						if index <= lengths[batch_ix]:
							# maximize the probability of choosing the correct action
							loss -= distribution.log_prob(target)

				# optimize the model
				generator.optimizer.zero_grad()
				loss.backward()
				generator.optimizer.step()
				generator.losses.append(loss.item())

	finally:
		# TESTING
		generator.eval()

		# with torch.no_grad():
		# 	# generate 10 Haikus
		# 	packed_haikus = generator.generate(batch_size=1, set_std=torch.full([1, dataset.embedding.embedding_dim], 1, dtype=torch.float32))

		# 	# unpack sequence
		# 	unpacked, lengths = pad_packed_sequence(packed_haikus, batch_first=True)
		# 	decoded = dataset.decode(unpacked)
		# 	for haiku in decoded:
		# 		print(haiku)

		# plot generator loss
		plt.title("Loss")
		plt.plot(generator.losses)
		plt.ylabel("Loss")
		plt.xlabel("Epochs")
		plt.savefig(f"{args.img_path}/generator_pretraining")
		plt.show()

	return generator

