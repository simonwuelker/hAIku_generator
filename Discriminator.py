import torch
import torch.nn as nn
import torch.optim as optim


class discriminator(nn.Module):
	def __init__(self, in_size, hidden_size=400, out_size=1, n_layers=1, lr=0.01):
		super(discriminator, self).__init__()

		self.in_size = in_size
		self.out_size = out_size
		self.lr = lr
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.chkpt_path = "models/Discriminator.pt"

		# LSTM Architecture
		self.lstm = nn.LSTM(in_size, hidden_size, n_layers, batch_first=True)
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

	def forward(self, input):
		batch_size = input.shape[0]
		self.reset_hidden(batch_size)

		lstm_out, self.hidden = self.lstm(input, self.hidden)
		output = self.network(lstm_out)
		return output[:, -1]  # return last value from every batch
		
	def learn(self, loss_d):
		self.optimizer.zero_grad()
		loss_d.backward()
		self.optimizer.step()	
		
	def loadModel(self):
		self.load_state_dict(torch.load(self.chkpt_path))

	def saveModel(self):
		torch.save(self.state_dict(), self.chkpt_path)

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), 
			torch.rand(self.n_layers, batch_size, self.hidden_size))
