import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class generator(nn.Module):
	def __init__(self, in_size, out_size, hidden_size=400, n_layers=2, lr=0.001):
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
		lstm_out, _ = self.lstm(input)
		output = self.network(lstm_out.reshape(-1, self.hidden_size))
		output = output.view_as(input)
		return output

	def generate(self, batch_size, seed=None):
		"""returns one generated sequence and optimizes the generator"""

		haiku_length = 2#np.random.randint(15, 20)  # length boundaries are arbitrary
		output = torch.zeros(batch_size, haiku_length, self.out_size)
		self.action_memory = torch.zeros(batch_size, haiku_length)

		# generate sequence starting from a given seed
		for i in range(haiku_length):
			# forward pass
			if i == 0:
				input = torch.zeros(batch_size, 1, self.out_size)
			else:
				input = output[:, :i].clone()  # inplace operation, clone is necessary
			probs = self(input)[:, -1].view(batch_size, self.out_size)

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
		batch_size = fake_sample.shape[0]
		seq_length = fake_sample.shape[1]

		# fill the reward memory using Monte Carlo
		self.reward_memory = torch.zeros_like(self.action_memory)
		for seq_ix in range(fake_sample.shape[1]): # the generator didnt take the first action, it was the seed

			#the amount of rollouts performed is proportional to their length
			num_rollouts = fake_sample.shape[1] - seq_ix
			qualities = torch.zeros(batch_size, num_rollouts)

			for rollout_ix in range(num_rollouts):
				# initiate starting sequence
				completed = torch.zeros_like(fake_sample)
				completed[:, :seq_ix] = fake_sample[:, :seq_ix]

				# rollout the remaining part of the sequence, rollout policy = generator policy
				for j in range(seq_ix, fake_sample.shape[1]):
					if j == 0:
						input = torch.zeros(batch_size, 1, self.out_size)
					else:
						input = completed[:, :j].view(batch_size, j, self.out_size)

					# choose action
					probs = self(input)[:, -1].view(batch_size, self.out_size)
					actions = torch.multinomial(probs, num_samples=1)

					# save action
					completed[:, j] = F.one_hot(actions, self.out_size) 

				# get the estimated reward for that rollout from the discriminator
				qualities[:, rollout_ix] = discriminator(completed).detach()

			self.reward_memory[:, seq_ix] = torch.mean(qualities, dim=1)

		# normalize the rewards
		# std, mean = torch.std_mean(self.reward_memory, dim=1, unbiased=False)  # avoid bessel
		# std[std == 0] = 1  # remove zeros from std
		# self.reward_memory = (self.reward_memory - mean.view(-1, 1)) / std.view(-1, 1)

		# calculate the discounted future rewards for every action
		discounted_rewards = torch.zeros(batch_size, seq_length)
		for batch_ix in range(batch_size):
			for seq_ix in range(seq_length):
				discounted_reward = 0
				for t in range(seq_ix, seq_length):
					discounted_reward += (self.discount**(t-seq_ix)) * self.reward_memory[batch_ix, t]
				discounted_rewards[batch_ix, seq_ix] = discounted_reward

		# calculate the loss using the REINFORCE Algorithm
		total_loss = 0
		for batch_ix in range(batch_size):
			for seq_ix in range(seq_length):
				total_loss += -1 * discounted_rewards[batch_ix, seq_ix] * self.action_memory[batch_ix, seq_ix]

		self.losses.append(total_loss.item())
		if total_loss.item() < 0:
			print("--")
			print(discounted_rewards, self.reward_memory)

		# optimize the agent
		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()
		return total_loss, self.action_memory, discounted_rewards, completed

	def loadModel(self, path=None):
		if path is None:
			path = self.pretrained_path
		self.load_state_dict(torch.load(path))

	def saveModel(self, path=None):
		if path is None:
			path = self.chkpt_path
		torch.save(self.state_dict(), path)
