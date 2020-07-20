# hAIku poem generator
import torch
import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
import numpy as np

import Generator
import Discriminator
import Tools
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import trange
import random


def example(haiku_length=8):
	"""returns one generated sequence and optimizes the generator"""
	generator.reset_hidden(batch_size=1)

	text = random.choice(list(dataset.unique_tokens))
	# generate the missing words to complete the haiku
	for i in range(haiku_length - len(text.split())):
		generator.reset_hidden(batch_size=1)  # every step is essentially a new forward pass

		input = torch.tensor([dataset.word_to_ix[word] for word in text.split()])
		outputs = generator(input.view(1, -1, 1))

		index = Tools.sample_from_output(outputs[-1])
		text = f"{text} {dataset.ix_to_word[index.item()]}"

	print(f"Haiku:{text}")

	# 	#total_loss += Tools.NLLLoss(probs, x, use_baseline = False)
	# 	total_loss += generator.criterion(probs, torch.argmax(target).unsqueeze(0))	#hiermit gehts
	# 	optimal[index, 0, torch.argmax(target)] = 1

	# #optimize generator
	# generator.optimizer.zero_grad()
	# total_loss.backward()
	# generator.optimizer.step()

	# generator.losses.append(total_loss.item())
	return dataset.encode(text).view(1, -1, 1)  # this doesnt support batch sizes > 1


modelsave_path = "models/"
batch_size = 1

torch.manual_seed(1)
np.random.seed(1)

dataset = Dataset(path="data/small_dataset.txt")
dataloader_ = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Init models
generator = Generator.generator(in_size=len(dataset.unique_tokens), out_size=len(dataset.unique_tokens))
discriminator = Discriminator.discriminator(in_size=len(dataset.unique_tokens))

# load models
generator.load_state_dict(torch.load("models/Generator_pretrained.pt"))
# discriminator.load_state_dict(torch.load("models/Discriminator.pt"))

# TRAINING
discriminator.train()
generator.train()

try:
	for epoch in trange(10):
		for real_sample in dataloader_:
			fake_sample = example()

			# take outputs from discriminator
			score_real = discriminator(real_sample)
			score_fake = discriminator(fake_sample)

			# Save scores for evaluation
			discriminator.scores_real.append(score_real.item())
			discriminator.scores_fake.append(score_fake.item())
			print(f"real:{score_real}, {Tools.decode(real_sample)}")
			print(f"fake:{score_fake}, {Tools.decode(fake_sample)}")

			# calculate loss
			loss_d = torch.mean(-torch.log(1 - score_fake) - torch.log(score_real))
			discriminator.losses.append(loss_d.item())

			# optimize discriminator
			discriminator.optimizer.zero_grad()
			loss_d.backward()
			discriminator.optimizer.step()

finally:
	# Models are always saved, even after a KeyboardInterrupt
	torch.save(discriminator.state_dict(), "models/Discriminator.pt")
	torch.save(generator.state_dict(), "models/Generator.pt")

	# TESTING
	discriminator.eval()
	generator.eval()

	with torch.no_grad():
		# run a lot of tests....
		pass

	# plot the graph of the different losses over time
	fig, ax = plt.subplots()
	# ax.plot(discriminator.scores_real, label = "Real")
	# ax.plot(discriminator.scores_fake, label = "Fake")
	ax.plot(generator.losses[2:], label="Generator Loss")
	plt.ylabel("Loss")
	plt.xlabel("training duration")
	ax.legend()

	# plt.show()
	print(generator.losses)
