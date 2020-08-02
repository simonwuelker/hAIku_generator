import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class OUActionNoise(object):
	def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
		self.theta = theta
		self.sigma = sigma
		self.mu = mu
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
	def __init__(self, state_shape, action_shape, memsize=100000):
		self.memsize = memsize
		self.mem_cntr = 0
		self.state_memory = np.empty((memsize, state_shape))
		self.action_memory = np.empty((memsize, action_shape))
		self.reward_memory = np.empty(memsize)
		self.new_state_memory = np.empty((memsize, state_shape))
		self.done_memory = np.empty(memsize)

	def store(self, state, action, reward, new_state, done):
		index = self.mem_cntr % self.memsize

		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.new_state_memory[index] = new_state
		self.done_memory[index] = done

		self.mem_cntr += 1

	def sample(self, batch_size):
		# prevent sampling parts of the buffer that have not yet been filled
		max_addr = min(self.memsize, self.mem_cntr)

		batch = np.random.choice(max_addr, batch_size)

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		new_states = self.new_state_memory[batch]
		done = self.new_state_memory[batch]

		return states, actions, rewards, new_states, done


class CriticNetwork(nn.Module):
	def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="tmp/ddpg"):
		super(CriticNetwork, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg")
		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

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

		self.action_value = nn.Linear(self.n_actions, fc2_dims)
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


class ActorNetwork(nn.Module):
	def __init__(self, alpha, in_size, fc1_dims, fc2_dims, embedding_dim, name, chkpt_dir="tmp/ddpg"):
		super(ActorNetwork, self).__init__()
		self.in_size = in_size
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.checkpoint_file = os.path.join(chkpt_dir, name + "_ddpg")

		self.embedding = nn.Embedding(self.in_size, self.embedding_dim)
		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
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
		self.mu = nn.Linear(self.fc2_dims, self.n_actions)
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


class generator():
	def __init__(self, alpha, beta, in_size, embedding_dim, fc1_dim=400, fc2_dim=300, batch_first=True):
		super(generator, self).__init__()

		self.in_size = in_size
		self.embedding_dim = embedding_dim
		self.losses = []
		# self.ReplayBuffer = ReplayBuffer(self.embedding_dim, self.out_size)
		# self.noise = OUActionNoise(mu=np.zeros(out_size))
		self.actor = ActorNetwork(alpha, in_size, fc1_dim, fc2_dim, embedding_dim, n_actions, "actor")
		self.critic = CriticNetwork(beta, in_size, fc1_dim, fc2_dim, embedding_dim, n_actions, "critic")
		self.target_actor = ActorNetwork(alpha, in_size, fc1_dim, fc2_dim, embedding_dim, n_actions, "target_actor")
		self.target_critic = CriticNetwork(beta, in_size, fc1_dim, fc2_dim, embedding_dim, n_actions, "target_critic")

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

	def reset_hidden(self, batch_size):
		self.hidden = (torch.rand(self.n_layers, batch_size, self.hidden_size), torch.rand(self.n_layers, batch_size, self.hidden_size))

	def save_models(self):
		print("Saving...")
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()
		self.target_actor.save_checkpoint()
		self.target_critic.save_checkpoint()

	def load_models(self):
		print("Loading...")
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()
		self.target_actor.load_checkpoint()
		self.target_critic.load_checkpoint()
