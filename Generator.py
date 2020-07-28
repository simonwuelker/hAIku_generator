import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer(object):
	def __init__(self, memsize=100000, state_shape, action_shape):
		self.memsize = memsize
		self.mem_cntr = 0
		self.state_memory = np.empty((memsize, state_shape))
		self.action_memory = np.empty((memsize, action_shape))
		self.reward_memory = np.empty(memsize)
		self.new_state_memory = np.empty((memsize, state_shape))
		self.done_memory = np.empty(memsize)

	def store(self, state, action, reward, new_state, done):
		index = mem_cntr % memsize

		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.new_state_memory[index] = new_state
		self.done_memory[index] = done

		mem_cntr += 1

	def sample(self, batch_size):
		# prevent sampling parts of the buffer that have not yet been filled
		max_addr = min(self.memsize, mem_cntr)

		batch = np.random.choice(max_addr, batch_size)

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		new_states = self.new_state_memory[batch]
		done = self.new_state_memory[batch]

		return states, actions, rewards, new_states, done

class generator(nn.Module):
	def __init__(self, in_size=80, hidden_size=400, n_layers=2, out_size=80, lr=0.04, embedding_dim=50, batch_first=True):
		super(generator, self).__init__()

		self.in_size = in_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.out_size = out_size
		self.embedding_dim = embedding_dim
		self.lr = lr
		self.losses = []
		self.ReplayBuffer = ReplayBuffer(self.embedding_dim, self.out_size)

		self.embedding = nn.Embedding(self.in_size, self.embedding_dim)
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.n_layers, batch_first=batch_first)

		self.network = nn.Sequential(
			nn.Linear(self.hidden_size, 400),
			nn.ReLU(),
			nn.Linear(400, 300),
			nn.ReLU(),
			nn.Linear(300, 128),
			nn.ReLU(),
			nn.Linear(128, out_size),
			nn.LogSoftmax(dim=1)
		)
		self.criterion = nn.NLLLoss()
		self.optimizer = optim.SGD(self.parameters(), lr)

	def forward(self, input):
		#das hier ist noch nicht auf batch first angepasst
		batch_size = input.shape[0]
		output = self.embedding(input.long()).view(batch_size, -1, self.embedding_dim)
		lstm_out, self.hidden = self.lstm(output, self.hidden)
		lstm_out = lstm_out.view([-1, self.hidden_size])
		output = self.network(lstm_out)
		return output

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))
