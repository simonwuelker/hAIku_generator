import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class CriticNetwork(nn.Module):
	def __init__(self, beta, embedding_dim, name, save_path="models/"):
		super(CriticNetwork, self).__init__()
		self.fc1_dims = 400
		self.fc2_dims = 300
		self.n_layers = 2
		self.embedding_dim = embedding_dim
		self.checkpoint_file = save_path + name		
		self.fc1 = nn.Linear(self.embedding_dim, self.fc1_dims)

		# This basically just speeds up convergence by initializing the lin layer
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
		self.bn1 = nn.LayerNorm(self.fc1_dims)

		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
		self.bn2 = nn.LayerNorm(self.fc2_dims)

		self.action_value = nn.Linear(self.embedding_dim, self.fc2_dims)
		f3 = 0.003
		self.q = nn.Linear(self.fc2_dims, 1)
		torch.nn.init.uniform_(self.fc2.weight.data, -f3, f3)
		torch.nn.init.uniform_(self.fc2.bias.data, -f3, f3)

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


class ActorNetwork(nn.Module):
	def __init__(self, alpha, embedding_dim, name, save_path="models/"):
		super(ActorNetwork, self).__init__()
		self.fc1_dims = 400
		self.fc2_dims = 300
		self.n_layers = 2
		self.embedding_dim = embedding_dim
		self.checkpoint_file = save_path + name

		self.fc1 = nn.Linear(self.embedding_dim, self.fc1_dims)
		f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
		torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
		self.bn1 = nn.LayerNorm(self.fc1_dims)

		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
		torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
		torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
		self.bn2 = nn.LayerNorm(self.fc2_dims)

		f3 = 0.003
		self.mu = nn.Linear(self.fc2_dims, self.embedding_dim)
		torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
		torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)


	def forward(self, state):
		x = self.fc1(state)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = torch.tanh(self.mu(x))
		return x

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.checkpoint_file))

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))


class Generator():
	def __init__(self, alpha, beta, embedding_dim, batch_first=True):
		super(Generator, self).__init__()
		self.embedding_dim = embedding_dim
		self.losses = []
		# define the core DDPG networks
		self.actor = ActorNetwork(alpha, embedding_dim, "actor")
		self.critic = CriticNetwork(beta, embedding_dim, "critic")
		self.target_critic = CriticNetwork(beta, embedding_dim, "target_critic")
		self.target_actor = ActorNetwork(alpha, embedding_dim, "target_actor")

	def __call__(self, input):
		"""
		Parameters:
			Input of shape[batch, sequence]
		"""
		action = self.actor(input)
		return action

	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = self.tau

		actor_params = self.actor.named_parameters()
		critic_params = self.critic.named_parameters()
		target_actor_params = self.target_actor.named_parameters()
		target_critic_params = self.target_critic.named_parameters()

		actor_state_dict = dict(actor_params)
		critic_state_dict = dict(critic_params)
		target_actor_state_dict = dict(target_actor_params)
		target_critic_state_dict = dict(target_critic_params)

		# Update Critic
		for name in critic_state_dict:
			critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()
		self.target_critic.load_state_dict(critic_state_dict)

		# Update Actor
		for name in actor_state_dict:
			actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()
		self.target_actor.load_state_dict(actor_state_dict)

	def save_models(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.target_critic.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()
		self.target_actor.load_checkpoint()
		self.target_critic.load_checkpoint()
