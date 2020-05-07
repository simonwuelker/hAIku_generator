import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import mido

import Converter

class generator(nn.Module):
	def __init__(self, in_size = 80, hidden_size = 64, n_layers = 2, out_size = 80):
		super(generator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.losses = []

		self.lstm = nn.LSTM(in_size, hidden_size, n_layers)
		self.hidden = (torch.rand(n_layers, 1, hidden_size), torch.rand(n_layers, 1, hidden_size))

		self.policy = nn.Sequential(
			nn.Linear(self.hidden_size, 128),
			nn.ReLU(),
			nn.Linear(128, 192),
			nn.ReLU(),
			nn.Linear(192, 128),
			nn.ReLU(),
			nn.Linear(128, 80)
			
			)
		self.optimizer = optim.Adam(self.parameters(), 0.05)
		self.criterion = nn.MSELoss()

	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input)
		lstm_out = lstm_out.view(lstm_out.shape[0]*lstm_out.shape[1], self.hidden_size)
		probs = self.policy(lstm_out)

		return probs

	def reset_hidden(self):
		self.hidden = (torch.rand(self.n_layers, 1, self.hidden_size), torch.rand(self.n_layers, 1, self.hidden_size))

	def play_example(self,length = 100, print_msg = False):
		array = self.generate_sequence()
		
		mid = Converter.toMidi(array)
		port = mido.open_output()

		for msg in mid.play():
			if print_msg:
				print(msg)
			port.send(msg)

	def generate_sequence(self, length = 100):
		self.reset_hidden()
		array = torch.empty([length, 80])

		state = torch.rand([80])

		for index in range(length):
			output = self(state.view(1, 1, 80))
			array[index] = output
			state = output

		return array

	def shift_state(self, state, new):
		#shift every notestate back by one index and delete the last one
		for index in range(len(state)-2, -1, -1):
			state[index+1] = state[index]

		#insert the new state at position 0
		state[0] = new
		return state