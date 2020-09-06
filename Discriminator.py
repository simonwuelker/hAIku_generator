# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
import torch
import torch.nn as nn
import torch.optim as optim


class discriminator(nn.Module):
	def __init__(self, in_size, hidden_size=600, out_size=1, n_layers=2, lr=0.01, batch_first=True, dropout=0.1):
		super(discriminator, self).__init__()

		self.in_size = in_size
		self.out_size = out_size
		self.lr = lr
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		# Architecture
		self.lstm = nn.LSTM(self.in_size, self.hidden_size, self.n_layers, batch_first=batch_first, dropout=dropout)
		self.network = nn.Sequential(
			nn.Linear(self.hidden_size, 800),
			nn.Dropout(0.1),
			nn.Linear(800, 600),
			nn.Dropout(0.1),
			nn.Linear(600, 100),
			nn.Dropout(0.1),
			nn.Linear(100, self.out_size),
			nn.Sigmoid()
			)

		self.criterion = nn.BCELoss()
		self.optimizer = optim.Adagrad(self.parameters(), self.lr)

		self.losses = []
		self.scores_real = []
		self.scores_fake = []
		self.pretrained_path = "models/Discriminator_pretrained.pt"
		self.save_path = "models/Discriminator.pt"

	def forward(self, input):
		batch_size = input.shape[0]

		lstm_out, _ = self.lstm(input)
		lstm_out = lstm_out.view(-1, self.hidden_size)
		scores = self.network(lstm_out)

		return scores.view(batch_size, -1)[:, -1]  # return last value from every batch

	def learn(self, loss_d):
		self.optimizer.zero_grad()
		loss_d.backward()
		self.optimizer.step()

	def loadModel(self, path=None):
		if path is None:
			path = self.pretrained_path
		self.load_state_dict(torch.load(path))

	def saveModel(self, path=None):
		if path is None:
			path = self.save_path
		torch.save(self.state_dict(), path)


	