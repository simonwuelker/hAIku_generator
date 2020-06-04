import torch
import torch.nn as nn
import torch.optim as optim

class discriminator(nn.Module):
	def __init__(self, in_size, sample_size, out_size = 1, lr = 0.01, batch_size = 1):
		super(discriminator, self).__init__()

		self.in_size = in_size
		self.out_size = out_size 
		self.lr = lr
		self.batch_size = batch_size
		self.sample_size = sample_size

		#CNN Architecture
		self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 3, kernel_size = 5)
		self.pool1 = nn.MaxPool3d(2, 2)
		self.conv2 = nn.Conv1d(in_channels = 3, out_channels = 9, kernel_size = 5)
		self.lin1 = nn.Linear(9 * (self.sample_size-8), 100)
		self.tan1 = nn.Tanh()
		self.lin2 = nn.Linear(100, out_size)
		self.sigmoid = nn.Sigmoid()

		self.optimizer = optim.Adagrad(self.parameters(), lr)

		self.losses = []
		self.scores_real = []
		self.scores_fake = []

		

	def forward(self, input):
		#BATCH SIZE!!
		input = input.view(1, 1, input.shape[0]).float()
		x = self.conv1(input)
		#x = self.pool1(x)
		x = self.conv2(x)
		x = x.flatten()
		x = self.lin1(x)
		x = self.tan1(x)
		x = self.lin2(x)
		output = self.sigmoid(x)
		return output