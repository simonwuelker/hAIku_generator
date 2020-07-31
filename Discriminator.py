import torch
import torch.nn as nn
import torch.optim as optim
import warnings


class discriminator(nn.Module):
	def __init__(self, in_size, hidden_size=400, out_size=1, n_layers=2, lr=0.01, embedding_dim=50, batch_first=True, dropout=0.3):
		super(discriminator, self).__init__()

		self.in_size = in_size
		self.out_size = out_size
		self.embedding_dim = embedding_dim
		self.lr = lr
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		# Architecture
		self.embedding = nn.Embedding(self.in_size, self.embedding_dim)
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.n_layers, batch_first=batch_first, dropout=dropout)
		self.lin1 = nn.Linear(self.hidden_size, 400)
		self.lin2 = nn.Linear(400, 300)
		self.lin3 = nn.Linear(300, 100)
		self.lin4 = nn.Linear(100, self.out_size)
		self.dropout = nn.Dropout(0.3)
		self.sigmoid = nn.Sigmoid()

		self.criterion = nn.BCELoss()
		self.optimizer = optim.Adagrad(self.parameters(), self.lr)

		self.losses = []
		self.scores_real = []
		self.scores_fake = []
		self.checkpoint_file = "models/Discriminator_pretrained.pt"

	def forward(self, input):
		batch_size = input.shape[0]
		self.zero_grad()
		self.reset_hidden(batch_size)

		embedded = self.embedding(input.long())
		lstm_out, self.hidden = self.lstm(embedded, self.hidden)
		lstm_out = lstm_out.view(-1, self.hidden_size)
		out1 = self.dropout(self.lin1(lstm_out))
		out2 = self.dropout(self.lin2(out1))
		out3 = self.dropout(self.lin3(out2))
		out = self.sigmoid(self.lin4(out3))

		return out.view(batch_size, -1)[:, -1]  # return last value from every batch

	def loadModel(self):
		try:
			self.load_state_dict(torch.load(self.checkpoint_file))
		except:
			warnings.warn("Failed to load Discriminator Model")

	def saveModel(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))

	