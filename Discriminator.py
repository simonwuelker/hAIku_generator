import torch
import torch.nn as nn
import torch.optim as optim

class discriminator(nn.Module):
	def __init__(self, in_size = 80, hidden_size = 50, out_size = 1, n_layers = 1, lr = 0.01, batch_size = 32):
		super(discriminator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.out_size = out_size 
		self.n_layers = n_layers
		self.lr = lr
		self.batch_size = batch_size
		self.hidden = (torch.rand(n_layers, batch_size, hidden_size), torch.rand(n_layers, batch_size, hidden_size))

		#Input ins LSTM hat dim3, dim0 ist die sequenz, dim1 ist der batch, dim2 sind die tatsächlichen daten
		self.lstm = nn.LSTM(in_size, hidden_size, n_layers)
		self.lin1 = nn.Linear(hidden_size, out_size)
		self.sigmoid = nn.Sigmoid()

		self.optimizer = optim.Adagrad(self.parameters(), lr)

		self.losses = []
		

	def forward(self, input):
		self.reset_hidden()
		lstm_out, self.hidden = self.lstm(input, self.hidden)
		lstm_out = lstm_out.view(lstm_out.shape[0]*lstm_out.shape[1], self.hidden_size)
		output = self.lin1(lstm_out)

		squeezed = self.sigmoid(output)
		return squeezed

	def reset_hidden(self):
		self.hidden = (torch.rand(self.n_layers, self.batch_size, self.hidden_size), torch.rand(self.n_layers, self.batch_size, self.hidden_size))
