import torch
import numpy as np


def predMaxReward(state, length, discriminator):
	"""predMaxReward(state) judges the quality of the given state by performing n rollouts"""
	batch_size = state.shape[1]
	simulations = 25  # increase simulations to increase reward accuracy
	maxScores = torch.tensor([float("-inf")] * batch_size)
	for roll_ix in range(simulations):
		completed = rollout(state, length)
		scores = discriminator(completed)  # only select the final judgement(when the whole sequence has been seen)
		scores = scores[-1, :]
		for batch_ix in range(batch_size):
			if scores[batch_ix] > maxScores[batch_ix]:
				maxScores[batch_ix] = scores[batch_ix]
	return maxScores


def expandTree(state, action):
	"""expandTree is the equivalent of the state transition function δ"""
	output = torch.zeros(state.shape[0], state.shape[1] + 1, state.shape[2])
	output[:, :state.shape[1]] = state

	# this can probably be simplified
	for batch_ix in range(state.shape[0]):
		output[batch_ix, -1] = action.detach()[batch_ix]
	return output


def Q(state, action, length, discriminator):
	"""
	Tries to estimate the quality of the action given the state via Monte Carlo
	Parameters:
		state:[batch_size, sequence_length, 1]
		action:[batch_size, 1]
	Returns:
		qualities:[batch_size, 1]
	"""
	print(state.shape)
	print(action)
	state = expandTree(state, action)
	print(state)
	assert False

	return quality.detach()


def rollout(input, length):
	"""takes a incomplete sequence and appends n random actions"""
	assert False
	output = torch.empty(input.shape[0], length, input.shape[2])
	output[:input.shape[0]] = input

	# randomly fill in the remaining values(rollout)
	for index in range(input.shape[0], length):
		batch = torch.zeros(input.shape[1], len(dataset.unique_tokens))
		for b in range(input.shape[1]):
			action = np.random.choice(np.arange(len(dataset.unique_tokens)))
			batch[b] = torch.zeros(len(dataset.unique_tokens))
			batch[b][action] = 1

		output[index] = batch
	assert False
	return output


def roundOutput(input):
	output = torch.zeros(input.shape)
	for batch_ix, batch in enumerate(input):
		output[batch_ix, torch.argmax(batch)] = 1
	return output


def NLLLoss(input, target, use_baseline=False):
	"""
	Computes the NLLLoss customized with a baseline

	Parameters:
		input:[batch_size, features]
		quality:[batch_size, features]
	Returns:
		[batch_size]
	"""
	assert input.shape == target.shape

	# target values are always positive, raising the baseline introduces negative rewards
	if use_baseline:
		baseline = torch.mean(target, dim=1)
		target = torch.sub(target, baseline.unsqueeze(1))

	# actually calculate the loss
	batch_size = input.shape[0]
	result = torch.empty(batch_size)
	for batch_ix in range(batch_size):
		result[batch_ix] = -1 * torch.dot(input[batch_ix], target[batch_ix])

	return torch.mean(result)


def sample_from_output(prob):
	"""samples one element from a given log probability distribution"""
	index = torch.multinomial(torch.exp(prob), num_samples=1)
	return index
