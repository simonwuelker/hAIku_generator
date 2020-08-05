import torch
import numpy as np

softm = torch.nn.Softmax(dim=1)
logSoftm = torch.nn.LogSoftmax(dim=1)


def betterSoftmax(input):
	"""calculates a better version of softmax which is more accurate but only works for positive numbers"""
	output = torch.empty(input.shape)
	for batch_ix, batch in enumerate(input):
		output[batch_ix] = batch / torch.sum(batch)
	return output


def predMaxReward(state, length, discriminator, dataset):
	"""predMaxReward(state) judges the quality of the given state by performing n rollouts"""
	batch_size = state.shape[1]
	simulations = 25  # increase simulations to increase reward accuracy
	maxScores = torch.tensor([float("-inf")] * batch_size)
	for roll_ix in range(simulations):
		completed = rolloutPartialSequence(state, length, dataset)
		scores = discriminator(completed)  # only select the final judgement(when the whole sequence has been seen)
		scores = scores[-1, :]
		for batch_ix in range(batch_size):
			if scores[batch_ix] > maxScores[batch_ix]:
				maxScores[batch_ix] = scores[batch_ix]
	return maxScores


def expandTree(state, action, dataset):
	"""expandTree is the equivalent of the state transition function δ"""
	num_classes = state.shape[2]
	batch_size = state.shape[1]
	seq_length = state.shape[0]
	output = torch.zeros(seq_length + 1, batch_size, num_classes)
	output[:seq_length] = state
	output[-1, :, action] = 1
	return output


def Q(state, length, discriminator, dataset):
	"""Q(state) returns the quality of all possible actions that can be performed in the given state"""
	batch_size = state.shape[1]
	# simulations = 2	# nr of actions to simulate
	# output = torch.zeros(simulations, batch_size)-1	# dimensions are wrong, will be transposed later
	# simulated = []	# keeps track of all the action that have been simulated
	# unique_simulations = 0	# counts all unique simulations

	# for _ in range(simulations):
	# 	#randomly sample one char based on the distribution from the dataset
	# 	action = np.random.choice(np.arange(len(dataset.unique_tokens)), p = distribution)
	# 	estimated_reward = predMaxReward(expandTree(state, action), length, discriminator)

	# 	#check if the action has already been simulated
	# 	if action in simulated:
	# 		index = simulated.index(action)
	# 		#action has beeen simulated twice so we need to check whether our new value is higher than the old one
	# 		if estimated_reward > output[index]:
	# 			output[index] = estimated_reward
	# 	else:
	# 		output[unique_simulations] = estimated_reward
	# 		simulated.append(action)
	# 		unique_simulations += 1

	# 	print(f"action nr.{_}: {action}")
	# print(output, unique_simulations)

	output = torch.zeros(28, batch_size)  # dimensions are wrong, will be transposed later
	for action in range(len(dataset.unique_tokens)):
		output[action] = predMaxReward(expandTree(state, action), length, discriminator, dataset)

	output = output.transpose(0, 1)

	return output  # simulated, output[:,:unique_simulations]


def rolloutPartialSequence(input, length, dataset):
	"""takes a incomplete sequence and appends n random actions"""
	output = torch.empty(length, input.shape[1], input.shape[2])
	output[:input.shape[0]] = input

	# randomly fill in the remaining values(rollout)
	for index in range(input.shape[0], length):
		batch = torch.zeros(input.shape[1], len(dataset.unique_tokens))
		for b in range(input.shape[1]):
			action = np.random.choice(np.arange(len(dataset.unique_tokens)), p=dataset.distribution)
			batch[b] = torch.zeros(len(dataset.unique_tokens))
			batch[b][action] = 1

		output[index] = batch
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
