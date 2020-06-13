import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import mido
import Tools

class generator(nn.Module):
	def __init__(self, in_size = 80, hidden_size = 400, n_layers = 2, out_size = 80, lr = 0.04):
		super(generator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.lr = lr
		self.losses = []

		self.lstm = nn.LSTM(in_size, hidden_size, n_layers)

		self.network = nn.Sequential(
			nn.Linear(self.hidden_size, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 128),
			nn.ReLU(),
			nn.Linear(128, out_size)
			
			)
		self.criterion = nn.NLLLoss()
		self.MSE = nn.MSELoss()
		self.optimizer = optim.Adam(self.parameters(), lr)


	def forward(self, input):
		assert input.shape[0] == 1	#sequence dimension might mess with the lin layer
		lstm_out, self.hidden = self.lstm(input, self.hidden)
		lstm_out = lstm_out.view(lstm_out.shape[0]*lstm_out.shape[1], self.hidden_size)
		output = self.network(lstm_out)
		return output

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))