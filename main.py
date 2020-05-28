#toller blogpost der erklärt wie der rl agent im generator funktioniert:http://karpathy.github.io/2016/05/31/rl/
#https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

import random
import os
import time

import Generator_LSTM
import Discriminator
import Tools

import matplotlib.pyplot as plt
import matplotlib.collections as collections

dataset_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/BW-KI-2020/notewise/"
modelsave_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/BW-KI-2020/models/"
load_models = False

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

sample_size = 100
dataloader = Tools.fetch_sample(sample_size, dataset_path)

# print("playing real sample:")
# Tools.playMidi(Tools.decode(next(dataloader).numpy()))
# print("playing fake sample:")
# Tools.playMidi(Tools.decode([random.randint(0,Tools.vocab_size-1) for _ in range(sample_size)]))
# assert False
#Init models
generator = Generator_LSTM.generator(in_size = Tools.vocab_size, out_size = Tools.vocab_size)
discriminator = Discriminator.discriminator(Tools.vocab_size, sample_size)


if load_models:
	generator.load_state_dict(torch.load(f"{modelsave_path}Generator_LSTM.pt"))
	discriminator.load_state_dict(torch.load(f"{modelsave_path}Discriminator.pt"))

#TRAINING
discriminator.train()
generator.train()

training_time = 10
start_time = time.time()
episode = 0

while time.time() < start_time + training_time:
	#print progress
	Tools.progressBar(start_time, time.time(), training_time, episode)
	episode += 1
	
	#load a random file to test the discriminator with
	filename = dataset_path + random.choice(os.listdir(dataset_path))

	real_sample = next(dataloader)
	fake_sample = torch.Tensor([random.randint(0,Tools.vocab_size) for _ in range(sample_size)])#.view(1, 1, 32)#generator(sample_size)

	#take outputs from discriminator and log them
	score_real = discriminator(real_sample)
	score_fake = discriminator(fake_sample)

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
# 	generator.play_example()

torch.save(discriminator.state_dict(), f"{modelsave_path}Discriminator.pt")
torch.save(generator.state_dict(), f"{modelsave_path}Generator_LSTM.pt")

#plot the graph of the different losses over time
fig, ax = plt.subplots()
# ax.plot(generator.losses, label = f"Generator")
# ax.plot(discriminator.losses, label = "Discriminator")
ax.plot(discriminator.scores_real, label = "Real")
ax.plot(discriminator.scores_fake, label = "Fake")
ax.legend()

plt.show()