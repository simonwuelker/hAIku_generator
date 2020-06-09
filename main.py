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
	
	haiku_length = np.random.randint(10, 17)	#length boundaries are arbitrary
	result = torch.zeros(haiku_length, batch_size, Tools.model.vector_size)
	seed = torch.rand(1, batch_size, Tools.model.vector_size)
	loss = 0

	for index in range(haiku_length):
		output = generator(seed)
		result[index, 0] = output
		best = Tools.bestAction(result[:index], haiku_length, discriminator.forward)
		loss += generator.criterion(output, best[0])

	#optimize generator
	generator.optimizer.zero_grad()
	loss.backward(retain_graph = True)
	generator.optimizer.step()
	print(Tools.decode(result))
	return Tools.roundOutput(result)

dataset_path = "dataset.txt"
modelsave_path = "models/"
load_models = True
batch_size = 1

torch.manual_seed(1)
np.random.seed(1)

dataloader = Tools.fetch_sample(dataset_path)

#Init models
generator = Generator.generator(in_size = Tools.model.vector_size, out_size = Tools.model.vector_size)
discriminator = Discriminator.discriminator(in_size = Tools.model.vector_size)


if load_models:
	generator.load_state_dict(torch.load(f"{modelsave_path}Generator.pt"))
	discriminator.load_state_dict(torch.load(f"{modelsave_path}Discriminator.pt"))

start_state = torch.zeros(Tools.model.vector_size)

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

	#calculate losses(Discriminator minimizes, generator maximizes)
	loss_d = torch.mean(-torch.log(1-score_fake) - torch.log(score_real))
	loss_g = torch.mean(torch.log(score_fake))

	#save losses
	generator.losses.append(loss_g.item())
	discriminator.losses.append(loss_d.item())

	#optimize discriminator
	discriminator.optimizer.zero_grad()
	loss_d.backward(retain_graph = True)
	discriminator.optimizer.step()
	
	#optimize generator
	# generator.optimizer.zero_grad()
	# loss_g.backward()
	# generator.optimizer.step()
	
#TESTING
discriminator.eval()
generator.eval()

# with torch.no_grad():
# 	print(f"Generator outputted: {Tools.decode(example())}")

torch.save(discriminator.state_dict(), f"{modelsave_path}Discriminator.pt")
torch.save(generator.state_dict(), f"{modelsave_path}Generator.pt")

#plot the graph of the different losses over time
fig, ax = plt.subplots()
# ax.plot(generator.losses, label = f"Generator")
# ax.plot(discriminator.losses, label = "Discriminator")
ax.plot(discriminator.scores_real, label = "Real")
ax.plot(discriminator.scores_fake, label = "Fake")
plt.ylabel("Loss")
plt.xlabel("training duration")
ax.legend()

plt.show()