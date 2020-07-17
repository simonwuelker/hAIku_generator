#hAIku poem generator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data	#cant inherit from torch.utils.data.Dataset otherwise
import numpy as np
import random

import Generator
import Discriminator
import Tools
from Dataset import Dataset

import matplotlib.pyplot as plt
from tqdm import trange

def example():
	"""returns one generated sequence and optimizes the generator"""
	generator.reset_hidden(batch_size = 1)
	
	haiku_length = 5#np.random.randint(8, 12)	#length boundaries are arbitrary
	output = torch.empty(haiku_length, batch_size, len(Tools.alphabet))
	previous_output = torch.rand(1, batch_size, len(Tools.alphabet))
	total_loss = 0

	optimal = torch.zeros(haiku_length, 1, len(Tools.alphabet))
	
	#generate sequence starting from a given seed
	text = "i"
	for i in range(haiku_length):
		generator.reset_hidden(batch_size = 1)	#every step is essentially a new forward pass

		input = torch.tensor([dataset.word_to_ix[word] for word in text.split()])
		outputs = generator(input.view(-1, 1, 1))
		index = Tools.sample_from_output(outputs[-1])
		text = f"{text} {dataset.ix_to_word[index.item()]}"

	# 	#total_loss += Tools.NLLLoss(probs, x, use_baseline = False)
	# 	total_loss += generator.criterion(probs, torch.argmax(target).unsqueeze(0))	#hiermit gehts
	# 	optimal[index, 0, torch.argmax(target)] = 1


	# #optimize generator
	# generator.optimizer.zero_grad()
	# total_loss.backward()
	# generator.optimizer.step()

	# generator.losses.append(total_loss.item())
	print(f"Output: {text}")
	return output.detach()

modelsave_path = "models/"
load_models = False
batch_size = 1
	
torch.manual_seed(1)
np.random.seed(1)

dataloader = Tools.fetch_sample("data/dataset.txt")

dataset = Dataset(path = "data/small_dataset.txt")
dataloader_ = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
print(dataset.unique_tokens)

#Init models
generator = Generator.generator(in_size = len(Tools.alphabet), out_size = len(Tools.alphabet))
discriminator = Discriminator.discriminator(in_size = len(Tools.alphabet))

if load_models:
	generator.load_state_dict(torch.load("models/Generator_pretrain.pt"))
	discriminator.load_state_dict(torch.load("models/Discriminator.pt"))

#TRAINING
discriminator.train()
generator.train()

try:
	for epoch in trange(10):
		for _, real_sample in dataloader_:
		real_sample = next(dataloader)
		print(next(dataloader))
		for element in dataloader_:
			print(element)
			assert False
		fake_sample = [example(), Tools.encode(["".join(random.choice(Tools.alphabet) for i in range(5))])][np.random.randint(2)]

		#take outputs from discriminator and log them
		score_real = discriminator(real_sample)
		score_fake = discriminator(fake_sample)

		#Save scores for evaluation
		discriminator.scores_real.append(score_real.item())
		discriminator.scores_fake.append(score_fake.item())
		print(f"real:{score_real}, {Tools.decode(real_sample)}")
		print(f"fake:{score_fake}, {Tools.decode(fake_sample)}")

		#calculate loss
		loss_d = torch.mean(-torch.log(1-score_fake) - torch.log(score_real))
		discriminator.losses.append(loss_d.item())

		#optimize discriminator
		discriminator.optimizer.zero_grad()
		loss_d.backward()
		discriminator.optimizer.step()
	
	#TESTING
	discriminator.eval()
	generator.eval()

	with torch.no_grad():
		pass

finally:
	#Models are always saved, even after a KeyboardInterrupt
	torch.save(discriminator.state_dict(), "models/Discriminator.pt")
	torch.save(generator.state_dict(), "models/Generator.pt")

	#plot the graph of the different losses over time
	fig, ax = plt.subplots()
	# ax.plot(discriminator.scores_real, label = "Real")
	# ax.plot(discriminator.scores_fake, label = "Fake")
	ax.plot(generator.losses[2:], label = "Generator Loss")
	plt.ylabel("Loss")
	plt.xlabel("training duration")
	ax.legend()

	# plt.show()
	print(generator.losses)