#genialer blogpost der erklärt wie der rl agent im generator funktioniert:http://karpathy.github.io/2016/05/31/rl/

import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

import random
import os

import Generator
import Discriminator

import matplotlib.pyplot as plt

#Gegebenenfalls GPU detecten
device = torch.device("cpu")
if torch.cuda.is_available():
	device = torch.device("cuda")

#Parameter
dataset_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/Datensatz/Numpy-Arrays/"
modelsave_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/Network/models/"
load_models = False

torch.manual_seed(1)

generator = Generator.generator()
discriminator = Discriminator.discriminator()

if load_models:
	generator.load_state_dict(torch.load(modelsave_path + "Generator.pt"))
	discriminator.load_state_dict(torch.load(modelsave_path + "Discriminator.pt"))
else:
	#generator.load_state_dict(torch.load(modelsave_path + "Generator_pretrained.pt"))
	pass



#TRAINING
discriminator.train()
generator.train()
for episode in range(100):
	print("Episode Nr.{}".format(episode))

	#load a random file to test the discriminator with
	filename = dataset_path + random.choice(os.listdir(dataset_path))
	loaded = torch.from_numpy(np.load(filename)).unsqueeze(1).float()

	#take output, calc loss and optimize discriminator
	score = discriminator(loaded)

	target = torch.ones(score.shape)
	loss = discriminator.criterion(score, target)
	discriminator.losses_real.append(loss.item())

	discriminator.optimizer.zero_grad()
	loss.backward()
	discriminator.optimizer.step()

	#Test and optimize Discriminator on a fake sequence by the generator
	sequence = generator.generate_sequence().unsqueeze(1)

	score = discriminator(sequence)
	target_d = torch.zeros(score.shape)

	loss_d = discriminator.criterion(score, target_d)
	discriminator.losses_fake.append(loss_d.item())

	discriminator.optimizer.zero_grad()
	loss_d.backward()
	discriminator.optimizer.step()


	#Test and optimize Generator
	sequence = generator.generate_sequence().unsqueeze(1)

	score = discriminator(sequence)
	target_g = torch.ones(score.shape)

	loss_g = generator.criterion(score, target_g)
	generator.losses.append(loss_g.item())

	generator.optimizer.zero_grad()
	loss_g.backward()
	generator.optimizer.step()
	
#TESTING
discriminator.eval()
generator.eval()
"""
#Discriminator mit einer random Datei testen
loaded = torch.from_numpy(np.load(dataset_path + random.choice(os.listdir(dataset_path)))).unsqueeze(1).float()
print(loaded.shape)
output = discriminator.forward(loaded)
print(output)
"""


torch.save(discriminator.state_dict(), modelsave_path + "Discriminator.pt")
torch.save(generator.state_dict(), modelsave_path + "Generator.pt")

#plot the graph of the different losses over time
fig, (sub1, sub2, sub3) =  plt.subplots(3, sharex = True)

sub1.plot(generator.losses)
sub1.set_title("Generator")

sub2.plot(discriminator.losses_real)
sub2.set_title("Discriminator real")

sub3.plot(discriminator.losses_fake)
sub3.set_title("Discriminator fake")

plt.show()


