#toller blogpost der erklärt wie der rl agent im generator funktioniert:http://karpathy.github.io/2016/05/31/rl/
#https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

import random
import os

import Generator_MCTS
import Discriminator
import MidiTools

import matplotlib.pyplot as plt
import matplotlib.collections as collections

dataset_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/Datensatz/notewise/"
modelsave_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/BW-KI-2020/models/"
load_models = True

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

sample_size = 32
dataloader = MidiTools.fetch_sample(sample_size, dataset_path)

#Init models
generator = Generator_MCTS.generator(sequence_length = sample_size, branching_factor = MidiTools.vocab_size)
discriminator = Discriminator.discriminator(in_size = MidiTools.vocab_size)


if load_models:
	generator.loadModel(f"{modelsave_path}Generator_MCTS.pt")
	discriminator.loadModel(f"{modelsave_path}Discriminator.pt")

#TRAINING
discriminator.train()

num_episodes = 100000

for episode in range(num_episodes):
	print("Episode Nr.{}".format(episode))
	
	#load a random file to test the discriminator with
	filename = dataset_path + random.choice(os.listdir(dataset_path))

	real_sample = next(dataloader)
	fake_sample = generator.next(sample_size)


	#take outputs from discriminator
	score_real = discriminator(real_sample)
	score_fake = discriminator(fake_sample)

	#calculate losses(Discriminator minimizes, generator maximizes)
	loss_d = torch.mean(-torch.log(1-score_fake) - torch.log(score_real))
	score_g = torch.mean(torch.log(score_fake))

	#save losses
	generator.losses.append(score_g.item())
	discriminator.losses.append(loss_d.item())


	#optimize discriminator
	# discriminator.optimizer.zero_grad()
	# loss_d.backward(retain_graph = True)
	# discriminator.optimizer.step()
	
	#optimize generator
	generator.optimize(score_g.item())
	
#TESTING
discriminator.eval()

discriminator.saveModel(f"{modelsave_path}Discriminator.pt")
generator.saveModel(f"{modelsave_path}Generator_MCTS.pt")

generator.save_example()

#plot the graph of the different losses over time
fig, ax = plt.subplots()
ax.plot(generator.losses, label = f"Generator")
ax.plot(discriminator.losses, label = "Discriminator")

ax.legend()

plt.show()