import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


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
			nn.LogSoftmax(dim=1)
		)
		self.criterion = nn.NLLLoss()
		self.optimizer = optim.Adam(self.parameters(), self.lr)

	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input, self.hidden)
		lstm_out = lstm_out.view([-1, self.hidden_size])
		output = self.network(lstm_out)
		return output


	def example(self, batch_size):
		"""returns one generated sequence and optimizes the generator"""
		# self.reset_hidden(batch_size)

		# haiku_length = np.random.randint(8, 12)  # length boundaries are arbitrary
		# output = torch.empty(batch_size, haiku_length, len(dataset.unique_tokens))

		# # generate sequence starting from a given seed
		# text = "i"
		# for i in range(haiku_length):
		# 	self.reset_hidden(batch_size=1)  # every step is essentially a new forward pass

		# 	input = torch.tensor([dataset.token_to_ix[word] for word in text.split()])
		# 	outputs = self(input)
		# 	index = Tools.sample_from_output(outputs[-1])
		# 	text = f"{text}{dataset.ix_to_token[index.item()]}"
		output = F.one_hot(torch.randint(28, size=(batch_size,10)), 28).float()
		return output

	def loadModel(self, path=None):
		if path is None:
			path = self.pretrained_path
		self.load_state_dict(torch.load(path))

	def saveModel(self):
		torch.save(self.state_dict(), self.chkpt_path)

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size),
						torch.rand(self.n_layers, batch_size, self.hidden_size))