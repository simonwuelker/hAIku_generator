import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import Tools
torch.autograd.set_detect_anomaly(True)


class generator(nn.Module):
	def __init__(self, in_size, out_size, hidden_size=400, n_layers=2, lr=0.005):
		super(generator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.lr = lr
		self.losses = []
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
		self.criterion = nn.MSELoss()
		self.optimizer = optim.Adam(self.parameters(), self.lr)

	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input, self.hidden)
		output = self.network(lstm_out)
		return output


	def generate(self, batch_size, seed=4):
		"""returns one generated sequence and optimizes the generator"""
		self.reset_hidden(batch_size)

		haiku_length = np.random.randint(15, 20)  # length boundaries are arbitrary
		output = torch.zeros(batch_size, haiku_length, self.out_size)
		self.last_probs = torch.zeros(batch_size, haiku_length, self.out_size)
		output[0, 0, seed] = 1

		# generate sequence starting from a given seed
		for i in range(haiku_length):
			self.reset_hidden(batch_size)  # every step is essentially a new forward pass

			# forward pass
			input = output[:, :i + 1].clone()  # inplace operation, clone is necessary
			probs = self(input)[:, -1]

			# choose action
			self.last_probs[:, i] = F.softmax(probs, dim=1)
			action = torch.multinomial(self.last_probs[:, i], num_samples=1)
			# encode action
			output[0, i] = F.one_hot(action, self.out_size)

		return output

	def learn(self, fake_sample, discriminator):
		#optimize the generator based on every character choice in the haiku
		batch_size = fake_sample.shape[0]
		loss = torch.zeros(fake_sample.shape[1])
		for index in range(fake_sample.shape[1]):
			seed = fake_sample[:, index]
			action_value_table = torch.zeros(batch_size, self.out_size)

			# simulate every action once
			for action in range(self.out_size):
				print(f"{index} - {action}")
				# initiate starting sequence
				completed = torch.zeros(fake_sample.shape)
				completed[:, :index] = seed

				# take the action
				completed[:, index, action] = 1

				# rollout the remaining part of the sequence, rolloutpolicy = generator policy
				for j in range(index + 1, fake_sample.shape[1]):
					self.reset_hidden(batch_size)
					# at the very start, there isnt really any input so just input zeros only
					if j == 0:
						input = torch.zeros(batch_size, 1, self.out_size)
					else:
						input = completed[:, :j].view(-1, j, self.out_size)
					# choose action
					probs = self(input)[:, -1]
					action_probs = torch.distributions.Categorical(probs)
					actions = action_probs.sample()
					# save action
					completed[:, j] = F.one_hot(actions, self.out_size) 
					

				# let the discriminator judge the complete sequence
				score = discriminator(completed)

				# save that score to the actionvalue table
				action_value_table[:, action] = score.detach()

			target = F.softmax(action_value_table, dim=1)
			loss[index] = self.criterion(self.last_probs[:, index], target)

		# optimize the generator
		total_loss = torch.mean(loss)
		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()

	def loadModel(self, path=None):
		if path is None:
			path = self.pretrained_path
		self.load_state_dict(torch.load(path))

	def saveModel(self, path=None):
		if path is None:
			path = self.chkpt_path
		torch.save(self.state_dict(), path)

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size),
						torch.rand(self.n_layers, batch_size, self.hidden_size))
