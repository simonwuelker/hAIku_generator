import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import MultivariateNormal
import numpy as np

class generator(nn.Module):
	def __init__(self, embedding_dim, model_path, hidden_size=400, n_layers=2):
		super(generator, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.losses = []
		self.discount = 0.9
		self.pretrained_path = f"{model_path}/Generator_pretrained.pt"
		self.trained_path = f"{model_path}/Generator.pt"

		self.lstm = nn.LSTM(self.embedding_dim + 1, self.hidden_size, self.n_layers, batch_first=True)
		self.mean = nn.Sequential(
			nn.Linear(self.hidden_size, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 128),
			nn.ReLU(),
			nn.Linear(128, embedding_dim),
			nn.Softmax(dim=1)
		)
		self.std = nn.Sequential(
			nn.Linear(self.hidden_size, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 128),
			nn.ReLU(),
			nn.Linear(128, embedding_dim),
			nn.Softmax(dim=1)
		)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters())


	def forward(self, sequence, lengths, std=None):
		"""
		Forwards the input through the model. If std is not None, 
		the model will only output the mean. std should be manually
		set during pretraining.
		"""
		batch_size = sequence.shape[0]
		seq_length = sequence.shape[1]

		# tell the model the remaining number of words at every step
		input = torch.zeros(batch_size, seq_length, self.embedding_dim + 1)
		for batch_ix in range(batch_size):
			input[:, :lengths[batch_ix], 0] = torch.arange(lengths[batch_ix], 0, -1)[:seq_length]
		input[:, :, 1:] = sequence

		# take the last values from the lstm-forward pass
		lstm_out, _ = self.lstm(input)
		lstm_out = lstm_out[:, -1].view(-1, self.hidden_size)

		# get the mean and standard deviation
		mean = self.mean(lstm_out)
		if std is None:
			std = torch.exp(self.std(lstm_out))

		# put the std on the diagonals of a m x m Matrix where m = embedding_dim
		# model only outputs variance values along the diagonals rn, might want 
		# to extend it onto the full covariance matrix later
		std_matrix = torch.diag_embed(std)

		# Construct a Multivariate Gaussian Distribution and sample an action
		# for pretraining, use log_prob since sample() doesnt provide gradient
		m = MultivariateNormal(mean, std_matrix)
		actions = m.sample()

		return actions, m

	def generate(self, batch_size, seed=None, set_std=None):
		"""
		Returns a batch of padded haikus as a PackedSequence Object.
		If the std for the normal distribution can optionally be provided through
		the set_std Parameter. If set to None, the model will generate the std.
		"""

		haiku_length = np.random.randint(12, 16)  # length boundaries are arbitrary
		output = torch.zeros(batch_size, haiku_length + 1, self.embedding_dim) # first element from the output is the inital seed
		self.action_memory = torch.zeros(batch_size, haiku_length)

		# generate sequence starting from a given seed
		for i in range(1, haiku_length + 1):
			# forward pass
			input = output[:, :i].clone()  # inplace operation, clone is necessary
			word, distribution = self(input, haiku_length, std=set_std)

			output[:, i] = word
			self.action_memory[:, i - 1] = distribution.log_prob(word)

		#remove the seed again
		output = output[:, 1:]

		# all haikus are initialized with max length, if a <eos> token is found the length is reduced
		haiku_lengths = torch.full([batch_size], haiku_length, dtype=torch.float32)

		# x = torch.tensor([[all(word_tensor) for word_tensor in batch] for batch in output == torch.zeros(self.embedding_dim)])
		# batch_indices, seq_indices = torch.nonzero(x, as_tuple=True)
		# for batch_ix, seq_ix in zip(batch_indices, seq_indices):
		# 	haiku_lengths[batch_ix] = seq_ix

		# pack the haikus into a PackedSequence object
		packed_output = pack_padded_sequence(output, haiku_lengths, batch_first=True)
		return packed_output

	def learn(self, fake_sample_packed, discriminator):
		fake_sample, lengths = pad_packed_sequence(fake_sample_packed, batch_first=True)
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
				# initiate starting sequence + seed
				completed = torch.zeros(batch_size, seq_length + 1, self.embedding_dim)
				completed[:, 1:seq_ix + 1] = fake_sample[:, :seq_ix]


				# rollout the remaining part of the sequence, rollout policy = generator policy
				for j in range(seq_ix + 1, fake_sample.shape[1] + 1):
					input_sequence = completed[:, :j].clone().view(batch_size, j, self.embedding_dim)

					# choose action 
					completed[:, j], _ = self(input_sequence, seq_length)

				# get the estimated reward for that rollout from the discriminator
				qualities[:, rollout_ix] = discriminator(completed[:, 1:]).detach()

			self.reward_memory[:, seq_ix] = torch.mean(qualities, dim=1)

		# normalize the rewards
		std, mean = torch.std_mean(self.reward_memory, dim=1, unbiased=False)  # avoid bessel
		std[std == 0] = 1  # remove zeros from std
		self.reward_memory = (self.reward_memory - mean.view(-1, 1)) / std.view(-1, 1)

		# create a discount vector [1, d, d^2, d^3, ...]
		discounted_rewards = torch.zeros(batch_size, seq_length)
		discounts = torch.full([seq_length], self.discount, dtype=torch.float32)
		discounts[0] = 1
		factors = torch.cumprod(discounts, dim=-1)

		# calculate the future discounted rewards
		for seq_ix in range(seq_length):
			discounted_rewards[:, seq_ix] = torch.matmul(self.reward_memory[:, seq_ix:], factors[:seq_length - seq_ix])


		# calculate the loss using the REINFORCE Algorithm
		total_loss = torch.sum(-1 * discounted_rewards * self.action_memory)

		# optimize the agent
		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		self.losses.append(total_loss.item())

	def loadModel(self, path=None):
		if path is None:
			path = self.pretrained_path
		self.load_state_dict(torch.load(path))

	def saveModel(self, path=None):
		if path is None:
			path = self.trained_path
		torch.save(self.state_dict(), path)