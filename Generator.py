import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import Tools

class generator(nn.Module):
	def __init__(self, in_size, out_size, hidden_size=400, n_layers=2, lr=0.0001):
		super(generator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.lr = lr
		self.losses = []
		self.discount = 0.9
		self.pretrained_path = "models/Generator_pretrained.pt"
		self.chkpt_path = "models/Generator.pt"

		self.lstm = nn.LSTM(self.in_size, self.hidden_size, self.n_layers, batch_first=True)

		self.network = nn.Sequential(
			nn.Linear(self.hidden_size, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 128),
			nn.ReLU(),
			nn.Linear(128, out_size),
			nn.Softmax(dim=1)
		)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters(), self.lr)

	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input, self.hidden)
		output = self.network(lstm_out.reshape(-1, self.hidden_size))
		output = F.relu(output)
		output = output.view(input.shape[0], -1, self.out_size)
		return output

	def generate(self, batch_size, seed=None):
		"""returns one generated sequence and optimizes the generator"""
		self.reset_hidden(batch_size)

		haiku_length = 5#np.random.randint(15, 20)  # length boundaries are arbitrary
		output = torch.zeros(batch_size, haiku_length, self.out_size)
		self.action_memory = torch.zeros(batch_size, haiku_length)
		if seed is not None:
			output[:, 0, seed] = 1

		# generate sequence starting from a given seed
		for i in range(1, haiku_length):
			self.reset_hidden(batch_size)  # every step is essentially a new forward pass

			# forward pass
			input = output[:, :i].clone()  # inplace operation, clone is necessary
			out = self(input)[:, -1].view(batch_size, self.out_size)

			# apply softmax without the e power thingy as relu has removed all negative values
			probs = torch.true_divide(out, torch.sum(out, dim=1).view(batch_size, 1))

			# sample an action and save the log probs of it
			actions = torch.multinomial(probs, num_samples=1).view(batch_size)
			log_probs = torch.zeros(batch_size)
			for batch in range(batch_size):
				log_probs[batch] = torch.log(probs[batch, actions[batch]])
			self.action_memory[:, i] = log_probs			
	
			# encode action
			output[:, i] = F.one_hot(actions, self.out_size)

		return output

	def learn(self, fake_sample, discriminator):
		# This is just plain REINFORCE

		# fill the reward memory using Monte Carlo
		self.reward_memory = torch.zeros_like(self.action_memory)
		for timestep in range(1, len(fake_sample)): # the generator didnt take the first action, it was the seed
			# initiate starting sequence
			completed = torch.zeros_like(fake_sample)
			completed[:, :timestep] = fake_sample[:, :timestep]

			# rollout the remaining part of the sequence, rolloutpolicy = generator policy
			for j in range(timestep + 1, fake_sample.shape[1]):
				self.reset_hidden(batch_size)
				input = completed[:, :j].view(-1, j, self.out_size)

				# choose action
				probs = self(input)[:, -1]
				actions = torch.multinomial(probs, num_samples=1)

				# save action
				completed[:, j] = F.one_hot(actions, self.out_size) 
				

			# get the estimated reward for that timestep from the discriminator HIER VLL TIMESTEP -1 KA
			self.reward_memory[timestep] = discriminator(completed)



		# batch_size = fake_sample.shape[0]
		# loss = torch.zeros(fake_sample.shape[1])
		# for index in range(fake_sample.shape[1]):
		# 	seed = fake_sample[:, index]
		# 	action_value_table = torch.zeros(batch_size, self.out_size)

		# 	# simulate every action once
		# 	for action in range(self.out_size):
		# 		print(f"{index} - {action}")
		# 		# initiate starting sequence
		# 		completed = torch.zeros(fake_sample.shape)
		# 		completed[:, :index] = seed

		# 		# take the action
		# 		completed[:, index, action] = 1

		# 		# rollout the remaining part of the sequence, rolloutpolicy = generator policy
		# 		for j in range(index + 1, fake_sample.shape[1]):
		# 			self.reset_hidden(batch_size)
		# 			# at the very start, there isnt really any input so just input zeros only
		# 			if j == 0:
		# 				input = torch.zeros(batch_size, 1, self.out_size)
		# 			else:
		# 				input = completed[:, :j].view(-1, j, self.out_size)
		# 			# choose action
		# 			probs = self(input)[:, -1]
		# 			action_probs = torch.distributions.Categorical(probs)
		# 			actions = action_probs.sample()
		# 			# save action
		# 			completed[:, j] = F.one_hot(actions, self.out_size) 
					

		# 		# let the discriminator judge the complete sequence
		# 		score = discriminator(completed)

		# 		# save that score to the actionvalue table
		# 		action_value_table[:, action] = score.detach()
		# 	# since there are no negative rewards, no power is needed in softmax
		# 	# ADD HERE
		# 	print(action_value_table.shape)
		# 	target = F.softmax(action_value_table, dim=1)
		# 	target = torch.true_divide
		# 	loss[index] = self.criterion(self.last_probs[:, index], target)

		# # optimize the generator
		# total_loss = torch.mean(loss)
		# self.optimizer.zero_grad()
		# total_loss.backward()
		# self.optimizer.step()

	def loadModel(self, path=None):
		if path is None:
			path = self.pretrained_path
		self.load_state_dict(torch.load(path))

	def saveModel(self, path=None):
		if path is None:
			path = self.chkpt_path
		torch.save(self.state_dict(), path)

	def reset_hidden(self, batch_size):
		self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size),
						torch.zeros(self.n_layers, batch_size, self.hidden_size))
