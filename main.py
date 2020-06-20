#hAIku poem generator
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import random

import Generator
import Discriminator
import Tools

import matplotlib.pyplot as plt
import time

def example():
	"""returns one generated sequence and optimizes the generator"""
	generator.reset_hidden(batch_size = 1)
	
	haiku_length = 5#np.random.randint(8, 12)	#length boundaries are arbitrary
	output = torch.empty(haiku_length, batch_size, len(Tools.alphabet))
	raw_output = torch.empty(haiku_length, batch_size, len(Tools.alphabet))
	previous_output = torch.rand(1, batch_size, len(Tools.alphabet))
	loss = 0

	optimal = torch.zeros(haiku_length, 1, len(Tools.alphabet))
	
	#generate sequence
	for index in range(haiku_length):
		probs = generator(previous_output)
		raw_output[index] = probs
		result = Tools.roundOutput(probs)
		output[index] = result
		previous_output = result.unsqueeze(0)
		target = Tools.Q(output[:index], haiku_length, discriminator)#Q function is slow as heck

		x = torch.zeros(target.shape)
		for batch in range(target.shape[0]):
			a = torch.zeros(target.shape[1])
			a[torch.argmax(target[batch])] = 1
			x[batch] = a

		# loss += Tools.NLLLoss(probs, x)
		# loss += generator.criterion(probs, target)	#hiermit gehts
		loss += generator.MSE(probs, target)
		# loss +=	Tools.NLLLoss(probs, torch.argmax(target).unsqueeze(0))
		# print(f"{Tools.NLLLoss_baseline(probs, x)}=={Tools.NLLLoss(probs, torch.argmax(target).unsqueeze(0))}")
		# assert False
		optimal[index, 0, torch.argmax(target)] = 1


	# loss = generator.MSE(raw_output, optimal)
	#optimize generator
	generator.optimizer.zero_grad()
	loss.backward()
	generator.optimizer.step()

	print(f"{Tools.decode(output)} Reward: {discriminator.forward(output).item()}")
	print(f"{Tools.decode(optimal)} Reward: {discriminator.forward(optimal).item()}")
	return output

dataset_path = "dataset.txt"
modelsave_path = "models/"
load_models = True
batch_size = 1

torch.manual_seed(1)
np.random.seed(1)

dataloader = Tools.fetch_sample(dataset_path)

#Init models
generator = Generator.generator(in_size = len(Tools.alphabet), out_size = len(Tools.alphabet))
discriminator = Discriminator.discriminator(in_size = len(Tools.alphabet))

if load_models:
	generator.load_state_dict(torch.load(f"{modelsave_path}Generator.pt"))
	discriminator.load_state_dict(torch.load(f"{modelsave_path}Discriminator.pt"))

# print(Tools.Q(Tools.encode([""]), 5, discriminator))
# print(Tools.Q(Tools.encode(["h"]), 5, discriminator))
# print(Tools.Q(Tools.encode(["ha"]), 5, discriminator))
# print(Tools.Q(Tools.encode(["hal"]), 5, discriminator))
# print(Tools.Q(Tools.encode(["hall"]), 5, discriminator))
# assert False

#TRAINING
discriminator.train()
generator.train()

training_time = 1000
start_time = time.time()
episode = 0
try:
	while time.time() < start_time + training_time:
		#print progress
		Tools.progressBar(start_time, time.time(), training_time, episode)
		episode += 1
		
		real_sample = next(dataloader)
		fake_sample = example()#Tools.encode(["".join(random.choice(Tools.alphabet) for i in range(5))])

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
	torch.save(discriminator.state_dict(), f"{modelsave_path}Discriminator.pt")
	torch.save(generator.state_dict(), f"{modelsave_path}Generator.pt")

	#plot the graph of the different losses over time
	fig, ax = plt.subplots()
	ax.plot(discriminator.scores_real, label = "Real")
	ax.plot(discriminator.scores_fake, label = "Fake")
	plt.ylabel("Scores")
	plt.xlabel("training duration")
	ax.legend()

	plt.show()

