import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import mido
import Tools

class generator(nn.Module):
	def __init__(self, in_size = 80, hidden_size = 128, n_layers = 2, out_size = 80, lr = 0.01):
		super(generator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.lr = lr
		self.losses = []

		self.lstm = nn.LSTM(in_size, hidden_size, n_layers)

		self.network = nn.Sequential(
			nn.Linear(self.hidden_size, 128),
			nn.ReLU(),
			nn.Linear(128, 192),
			nn.ReLU(),
			nn.Linear(192, 128),
			nn.ReLU(),
			nn.Linear(128, out_size),
			nn.Softmax(dim = 0)
			
			)
		self.criterion = nn.NLLLoss()
		self.optimizer = optim.Adam(self.parameters(), lr)


	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input, self.hidden)
		output = self.network(lstm_out)
		return output

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))