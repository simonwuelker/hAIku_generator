import torch
import torch.nn as nn
import torch.optim as optim

class discriminator(nn.Module):
	def __init__(self, in_size, hidden_size = 400, out_size = 1, n_layers = 1,lr = 0.01, batch_size = 1):
		super(discriminator, self).__init__()

		self.in_size = in_size
		self.out_size = out_size 
		self.lr = lr
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		#LSTM Architecture
		self.lstm = nn.LSTM(in_size, hidden_size, n_layers)
		self.network = nn.Sequential(
			nn.Linear(self.hidden_size, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 100),
			nn.ReLU(),
			nn.Linear(100, 1),
			nn.Sigmoid()
			
		)
		self.criterion = nn.BCELoss()
		self.optimizer = optim.Adagrad(self.parameters(), lr)

		self.losses = []
		self.scores_real = []
		self.scores_fake = []

	def reset_hidden(self, batch_size):
		#discriminator hidden is not random
		self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size), torch.zeros(self.n_layers, batch_size, self.hidden_size))
	
	def forward(self, input):
		batch_size = input.shape[1]
		assert batch_size == 1	#return fails with batch size > 1
		self.reset_hidden(batch_size)

		lstm_out, self.hidden = self.lstm(input, self.hidden)
		output = self.network(lstm_out)
		return torch.mean(output).view(1,1,1)	#return the last score, incomplete sequence may be judged incorrectly