#hAIku poem generator
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

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
	
	#generate sequence
	total_loss = 0
	for index in range(haiku_length):
		probs = generator(previous_output)
		raw_output[index] = probs
		result = Tools.roundOutput(probs)
		output[index] = result
		previous_output = result.unsqueeze(0)
		target = Tools.Q(output[:index], haiku_length, discriminator)#Q function is slow as heck

		total_loss += generator.MSE(probs, target)

	#optimize generator
	generator.optimizer.zero_grad()
	total_loss.backward()
	generator.optimizer.step()

	print(f"{Tools.decode(output)} Reward: {discriminator.forward(output).item()}")
	return output

dataset_path = "dataset.txt"
modelsave_path = "models/"
load_models = False
batch_size = 1

torch.manual_seed(1)
np.random.seed(1)

dataloader = Tools.fetch_sample(dataset_path)

#Init models
generator = Generator.generator(in_size = Tools.word2vec_model.vector_size, out_size = Tools.word2vec_model.vector_size)
discriminator = Discriminator.discriminator(in_size = Tools.word2vec_model.vector_size)

if load_models:
	generator.load_state_dict(torch.load(f"{modelsave_path}Generator.pt"))
	discriminator.load_state_dict(torch.load(f"{modelsave_path}Discriminator.pt"))

start_state = torch.zeros(Tools.word2vec_model.vector_size)

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
		fake_sample = example()

		#take outputs from discriminator and log them
		score_real = discriminator(real_sample)
		score_fake = discriminator(fake_sample)

		#Save scores for evaluation
		discriminator.scores_real.append(score_real.item())
		discriminator.scores_fake.append(score_fake.item())
		print(score_real, 1)
		print(score_fake, 1)

		#calculate losses(Discriminator minimizes, generator maximizes
		loss_d = torch.mean(-torch.log(1-score_fake) - torch.log(score_real))

		#save losses
		discriminator.losses.append(loss_d.item())

		#optimize discriminator
		discriminator.optimizer.zero_grad()
		loss_d.backward()
		discriminator.optimizer.step()

finally:	
	#TESTING
	discriminator.eval()
	generator.eval()

	#test without operation tracking to save memory
	with torch.no_grad():
		pass

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
