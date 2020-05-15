#toller blogpost der erklärt wie der rl agent im generator funktioniert:http://karpathy.github.io/2016/05/31/rl/
#https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b
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

def fetch_sample(length, batch_size = 32):
	while True:
		filename = dataset_path + random.choice(os.listdir(dataset_path))
		sample = torch.from_numpy(np.load(filename))
		#das ist jetzt hochverwirrend
		array = torch.empty(length, batch_size)
		for i in range(batch_size)
			
		for i in range(int(len(sample)/length)):
			yield sample[i*length:(i+1)*length].unsqueeze(1).float()


#Parameter
dataset_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/Datensatz/Numpy-Arrays/"
modelsave_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/Network/models/"
load_models = False

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

generator = Generator.generator()
discriminator = Discriminator.discriminator()

sample_size = 100
dataloader = fetch_sample(sample_size)

if load_models:
	generator.load_state_dict(torch.load(modelsave_path + "Generator.pt"))
	discriminator.load_state_dict(torch.load(modelsave_path + "Discriminator.pt"))
else:
	#generator.load_state_dict(torch.load(modelsave_path + "Generator_pretrained.pt"))
	pass



#TRAINING
discriminator.train()
generator.train()

for episode in range(1000):
	print("Episode Nr.{}".format(episode))
	
	#load a random file to test the discriminator with
	filename = dataset_path + random.choice(os.listdir(dataset_path))

	real_sample = next(dataloader)
	fake_sample = generator.generate_sequence(sample_size)#.unsqueeze(1)

	if len(real_sample) == 0:
		print("error filename {}".format(filename))
		continue

	#take outputs from discriminator
	score_real = discriminator(real_sample)
	score_fake = discriminator(fake_sample)

	#calculate losses
	loss_d = torch.mean(-torch.log(1-score_fake) - torch.log(score_real))
	loss_g = torch.mean(-torch.log(score_fake))

	#save losses
	generator.losses.append(loss_g.item())
	discriminator.losses.append(loss_d.item())


	#optimize discriminator if his loss is above 0.25, otherwise let the generator exploit him
	if loss_d > 0.25:
		discriminator.optimizer.zero_grad()
		loss_d.backward(retain_graph = True)
		discriminator.optimizer.step()
	else:
		print("Not training Discriminator")

	
	#optimize generator
	generator.optimizer.zero_grad()
	loss_g.backward()
	generator.optimizer.step()
	
#TESTING
discriminator.eval()
generator.eval()

torch.save(discriminator.state_dict(), modelsave_path + "Discriminator.pt")
torch.save(generator.state_dict(), modelsave_path + "Generator.pt")

generator.save_example()

#plot the graph of the different losses over time
plt.plot(generator.losses, label = "Generator")
plt.plot(discriminator.losses, label = "Discriminator")
plt.legend()

plt.show()
