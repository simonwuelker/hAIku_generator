import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np

import Discriminator
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def generate_random(dataset, batch_size):
	"""
	Generates a collection of random haikus(batch_size), padds them to equal length
	and returns them as a PackedSequenceObject
	"""
	lengths = []
	unpadded_data = []

	for _ in range(batch_size):
		# generate a single fake sample
		fake_length = random.randint(8, 13)
		fake_sample = " ".join([random.choice(tuple(dataset.word2vec.vocab.keys())) for word in range(fake_length)])
		fake_sample = dataset.encode(fake_sample)
		unpadded_data.append(fake_sample)
		lengths.append(fake_length)

	# padd and pack the fake samples
	padded_data = pad_sequence(unpadded_data, batch_first=True)
	packed_data = pack_padded_sequence(padded_data, lengths, batch_first=True, enforce_sorted=False)
	return packed_data

def train(discriminator, dataset, args):
	# TRAINING
	batch_size = int(args.pretrain_dis[0])
	epochs = int(args.pretrain_dis[1])
	test_episodes = 2000

	training_iterator = dataset.DataLoader(len(dataset) - test_episodes, batch_size=batch_size)
	testing_iterator = dataset.DataLoader(start=len(dataset) - test_episodes, end=len(dataset), batch_size=batch_size)
	training_progress = tqdm(total=len(dataset) * epochs, desc="Training")
	discriminator.train()
	try:
		for epoch in range(epochs):
			for real_sample in training_iterator:
				fake_sample = generate_random(dataset, batch_size)

				# update training prorgess bar
				training_progress.update(batch_size)

				# Pass the samples through the discriminator
				score_real = discriminator(real_sample)
				score_fake = discriminator(fake_sample)

				# optimize
				loss = torch.mean(- torch.log(0.001 + score_real) - torch.log(1.001 - score_fake))
				discriminator.learn(loss)

				# save results
				discriminator.scores_real.append(score_real.mean().item())
				discriminator.scores_fake.append(score_fake.mean().item())
				discriminator.losses.append(loss.item())

	finally:
		# TESTING
		testing_progress = tqdm(total=test_episodes, desc="Testing")
		discriminator.eval()

		with torch.no_grad():
			real_scores = torch.zeros(test_episodes, batch_size)
			fake_scores = torch.zeros(test_episodes, batch_size)
			for index, real_sample in enumerate(testing_iterator):
				# update progress bar
				testing_progress.update(batch_size)

				fake_sample = generate_random(dataset, batch_size)

				#forward pass
				real_scores[index] = discriminator(real_sample).view(batch_size)
				fake_scores[index] = discriminator(fake_sample).view(batch_size)


			print(f"The mean score for real samples from the training set is: {torch.mean(real_scores)}")
			print(f"The mean score for fake samples is: {torch.mean(fake_scores)}")
			print(f"{round((torch.sum(real_scores > 0.5).item()/(test_episodes * batch_size))*100, 2)}% of real samples were classified correctly")
			print(f"{round((torch.sum(fake_scores < 0.5).item()/(test_episodes * batch_size))*100, 2)}% of fake samples were classified correctly")


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
		plt.savefig(f"{args.img_path}/discriminator_pretraining.png")
		plt.show()

	return discriminator