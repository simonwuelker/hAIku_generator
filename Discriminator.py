import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class discriminator(nn.Module):
	def __init__(self, in_size, model_path, hidden_size=600, out_size=1, n_layers=2, dropout=0.1):
		super(discriminator, self).__init__()

		# Parameters
		self.in_size = in_size
		self.out_size = out_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		# Architecture
		self.lstm = nn.LSTM(self.in_size, self.hidden_size, self.n_layers, batch_first=True, dropout=dropout, bidirectional=True)
		self.network = nn.Sequential(
			nn.Linear(self.hidden_size * 2, 800),
			nn.Dropout(dropout),
			nn.Linear(800, 600),
			nn.Dropout(dropout),
			nn.Linear(600, 100),
			nn.Dropout(dropout),
			nn.Linear(100, self.out_size),
			nn.Sigmoid()
			)

		self.criterion = nn.BCELoss()
		self.optimizer = optim.Adagrad(self.parameters())

		self.losses = []
		self.scores_real = []
		self.scores_fake = []
		self.pretrained_path = f"{model_path}/Discriminator_pretrained.pt"
		self.save_path = f"{model_path}/Discriminator.pt"

	def forward(self, input):
		"""
		Forwards the input through the model

		Parameters:
				input(Packedsequence):the input
		Returns:
				scores(Tensor):A Tensor of Shape [batch_size, 1] containing values [0, 1] that describe how real 
							   the discriminator thinks the samples from the input are.
		"""
		lstm_out, _ = self.lstm(input)

		if isinstance(lstm_out, PackedSequence):
			#unpack the output
			lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

		last_hidden = lstm_out[:, -1]  # take the last hidden states from every batch
		scores = self.network(last_hidden)

		return scores

	def learn(self, loss_d):
		"""Optimizes the discriminator on the loss provided"""
		self.optimizer.zero_grad()
		loss_d.backward()
		self.optimizer.step()

	def loadModel(self, path=None):
		"""Load the model from a path that can optionally be provided"""
		if path is None:
			path = self.pretrained_path
		self.load_state_dict(torch.load(path))

	def saveModel(self, path=None):
		"""Saves the model to a path that can optionally be provided"""
		if path is None:
			path = self.save_path
		torch.save(self.state_dict(), path)


	