import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import mido

import Converter

class generator(nn.Module):
	def __init__(self, in_size = 80, hidden_size = 128, n_layers = 2, out_size = 80, lr = 0.01, batch_size = 1):
		super(generator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.lr = lr
		self.batch_size = batch_size
		self.losses = []

		self.lstm = nn.LSTM(in_size, hidden_size, n_layers)
		self.hidden = (torch.rand(n_layers, batch_size, hidden_size), torch.rand(n_layers, batch_size, hidden_size))

		self.policy = nn.Sequential(
			nn.Linear(self.hidden_size, 128),
			nn.ReLU(),
			nn.Linear(128, 192),
			nn.ReLU(),
			nn.Linear(192, 128),
			nn.ReLU(),
			nn.Linear(128, in_size),
			nn.Softmax()
			
			)
		self.optimizer = optim.Adam(self.parameters(), lr)


	def forward(self, input):
		lstm_out, self.hidden = self.lstm(input)
		lstm_out = lstm_out.view(self.hidden_size)#view geändert
		probs = self.policy(lstm_out)

		return probs

	def reset_hidden(self):
		self.hidden = (torch.rand(self.n_layers, self.batch_size, self.hidden_size), torch.rand(self.n_layers, self.batch_size, self.hidden_size))

	def play_example(self,length = 100, print_msg = False):
		array = self.generate_sequence(length)
		
		mid = Converter.toMidi(array)
		port = mido.open_output()

		for msg in mid.play():
			if print_msg:
				print(msg)
			port.send(msg)

	def save_example(self, length = 100):
		array = self.generate_sequence(length)
		output = np.empty(array.shape[0])
		for index, element in enumerate(array):
			output[index] = torch.argmax(element).item()

		mid = Converter.decode(output)
		mid.save("output.mid")

	def generate_sequence(self, length):
		self.reset_hidden()
		array = torch.empty([length, self.batch_size, self.in_size])

		state = torch.rand([self.batch_size, self.in_size])

		for index in range(length):
			output = self(state.view(1, self.batch_size, self.in_size))

			array[index] = output
			state = output

		return array

