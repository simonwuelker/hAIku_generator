#toller blogpost der erklärt wie der rl agent im generator funktioniert:http://karpathy.github.io/2016/05/31/rl/
#https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b
#hAIku poem generator
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

import Generator
import Discriminator
import Tools

import matplotlib.pyplot as plt
import matplotlib.collections as collections
import time

def example():
	generator.reset_hidden(batch_size = 1)
	
	haiku_length = 10#np.random.randint(8, 12)	#length boundaries are arbitrary
	seed = torch.empty(haiku_length, batch_size, Tools.word2vec_model.vector_size)
	single = torch.rand(batch_size, Tools.word2vec_model.vector_size)
	for i in range(haiku_length):
		seed[i] = single 
	loss = 0
	output = generator(seed)

	optimal_haiku = torch.empty(haiku_length, 1, Tools.word2vec_model.vector_size)

	for index in range(haiku_length):
		best = Tools.bestAction(output[:index], haiku_length, discriminator.forward)
		loss += generator.criterion(output[index, 0], best[0])

	# #optimize generator
	generator.optimizer.zero_grad()
	loss.backward(retain_graph = True)
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
	loss_d = torch.mean(-torch.log(1.1-score_fake) - torch.log(score_real))
	loss_g = torch.mean(torch.log(score_fake))

	#save losses
	generator.losses.append(loss_g.item())
	discriminator.losses.append(loss_d.item())

	#optimize discriminator
	discriminator.optimizer.zero_grad()
	loss_d.backward()
	discriminator.optimizer.step()
	
#TESTING
discriminator.eval()
generator.eval()

# with torch.no_grad():
# 	print(Tools.predMaxReward(Tools.encode("concentrate concentrate"), 2, discriminator.forward))

torch.save(discriminator.state_dict(), f"{modelsave_path}Discriminator.pt")
torch.save(generator.state_dict(), f"{modelsave_path}Generator.pt")

#plot the graph of the different losses over time
fig, ax = plt.subplots()
# ax.plot(generator.losses, label = f"Generator")
# ax.plot(discriminator.losses, label = "Discriminator")
ax.plot(discriminator.scores_real, label = "Real")
ax.plot(discriminator.scores_fake, label = "Fake")
plt.ylabel("Scores")
plt.xlabel("training duration")
ax.legend()

plt.show()
print(discriminator.scores_real)
print(discriminator.scores_fake)
