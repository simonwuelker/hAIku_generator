import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import Tools

class generator(nn.Module):
	def __init__(self, in_size = 80, hidden_size = 400, n_layers = 2, out_size = 80, lr = 0.04, embedding_dim=50):
		super(generator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.embedding_dim = embedding_dim
		self.lr = lr
		self.losses = []

		self.embedding = nn.Embedding(self.in_size, self.embedding_dim)
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.n_layers)

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
		self.optimizer = optim.SGD(self.parameters(), lr)


	def forward(self, input):
		seq_length = input.shape[0]
		output = self.embedding(input.long()).view(seq_length, -1, self.embedding_dim)
		lstm_out, self.hidden = self.lstm(output, self.hidden)
		lstm_out = lstm_out.view([-1, self.hidden_size])
		output = self.network(lstm_out)
		return output

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))