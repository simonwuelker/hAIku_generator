import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


class Critic(nn.Module):
	def __init__(self, beta, n_actions, name, save_path="models/"):
		super(Critic, self).__init__()
		self.fc1_dims = 400
		self.fc2_dims = 300
		self.n_layers = 2
		self.n_actions = n_actions
		self.checkpoint_file = save_path + name

		self.fc1 = nn.Linear(self.n_actions, self.fc1_dims)
		self.bn1 = nn.LayerNorm(self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.bn2 = nn.LayerNorm(self.fc2_dims)
		self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
		self.q = nn.Linear(self.fc2_dims, 1)

		self.optimizer = optim.Adam(self.parameters(), lr=beta)

	def forward(self, state, action):
		input = torch.cat((state, action))
		state_value = self.fc1(state)
		state_value = self.bn1(state_value)
		state_value = F.relu(state_value)
		state_value = self.fc2(state_value)
		state_value = self.bn2(state_value)

		action_value = F.relu(self.action_value(action))
		state_action_value = F.relu(torch.add(state_value, action_value))
		state_action_value = self.q(state_action_value)

		return state_action_value

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.checkpoint_file))

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))


class Actor(nn.Module):
	def __init__(self, alpha, n_actions, name, save_path="models/"):
		super(Actor, self).__init__()
		self.fc1_dims = 400
		self.fc2_dims = 300
		self.hidden_size = n_actions
		self.n_layers = 2
		self.n_actions = n_actions
		self.checkpoint_file = save_path + name

		self.lstm = nn.LSTM(input_size=n_actions, hidden_size=n_actions, num_layers=2, batch_first=True)
		self.mu = nn.Linear(self.hidden_size, self.n_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)


	def forward(self, state):
		lstm_out, self.hidden = self.lstm(state, self.hidden)
		# take the last value from every batch
		lstm_last = lstm_out[:, -1]
		actions = self.mu(lstm_last)
		return actions

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.checkpoint_file))

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))


class generator():
	def __init__(self, lr_actor, lr_critic, n_actions):
		super(generator, self).__init__()
		self.n_actions = n_actions
		self.losses = []
		self.training = True # Noise is only applied during training
		self.tau = 0.1

		# define the core DDPG networks
		self.actor = Actor(lr_actor, n_actions, "actor")
		self.critic = Critic(lr_critic, n_actions, "critic")
		self.target_actor = Actor(lr_actor, n_actions, "target_actor")
		self.target_critic = Critic(lr_critic, n_actions, "target_critic")

		# initialize target network parameters
		for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
			target_param.data.copy_(param.data)

		for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
			target_param.data.copy_(param.data)

	def generate(self, dataset, batch_size):
		haiku_length = random.randint(7, 10)
		result = torch.zeros(batch_size, haiku_length, self.n_actions)

		# initiate seed
		seed = random.choices(list(dataset.model.wv.vocab.keys()), k=batch_size)
		for index, word in enumerate(seed):
			result[index, 0] = torch.tensor(np.copy(dataset.model[word]))

		# generate the rest of the haiku
		for index in range(1, haiku_length):
			self.actor.reset_hidden(batch_size=batch_size)
			input = result[:, :index]
			output = self.actor(input)
			result[:, index] = output

		return result

	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = self.tau

		# update target networks 
		for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
			target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

	def saveModels(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.target_critic.save_checkpoint()

	def loadModels(self):
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()
		self.target_actor.load_checkpoint()
		self.target_critic.load_checkpoint()

	def train(self):
		# maybe set state for actor/critic network as well
		self.actor.train()
		self.critic.train()
		self.target_actor.train()
		self.target_critic.train()

	def eval(self):
		self.actor.eval()
		self.critic.eval()
		self.target_actor.eval()
		self.target_critic.eval()


