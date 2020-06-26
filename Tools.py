import torch
import numpy as np
import string
import sys
import random

alphabet = "halo"#string.ascii_lowercase + "," + " "
char_to_ix = {char:ix for ix, char in enumerate(alphabet)}
ix_to_char = {ix:char for ix, char in enumerate(alphabet)}
distribution = np.array([0.2, 0.2, 0.4, 0.2])#np.load("distribution.npy")
softm = torch.nn.Softmax(dim = 1)
logSoftm = torch.nn.LogSoftmax(dim=1)

def betterSoftmax(input):
	"""calculates a better version of softmax which is more accurate but only works for positive numbers"""
	output = torch.empty(input.shape)
	for batch_ix, batch in enumerate(input):
		output[batch_ix] = batch/torch.sum(batch)
	return output

def predMaxReward(state, length, discriminator):
	"""predMaxReward(state) judges the quality of the given state by performing n rollouts"""
	batch_size = state.shape[1]
	simulations = 25	#increase simulations to increase reward accuracy
	maxScores = torch.tensor([float("-inf")]*batch_size)
	for roll_ix in range(simulations):
		completed = rolloutPartialSequence(state, length)
		scores = discriminator(completed)#only select the final judgement(when the whole sequence has been seen)
		scores = scores[-1,:]
		for batch_ix in range(batch_size):
			if scores[batch_ix] > maxScores[batch_ix]:
				maxScores[batch_ix] = scores[batch_ix]
	return maxScores


def expandTree(state, action):
	"""expandTree is the equivalent of the state transition function δ"""
	batch_size = state.shape[1]
	seq_length = state.shape[0]
	output = torch.zeros(seq_length+1, batch_size, len(alphabet))
	output[:seq_length] = state
	output[-1, :, action] = 1
	return output

def Q(state, length, discriminator):
	"""Q(state) returns the quality of all possible actions that can be performed in the given state"""
	batch_size = state.shape[1]
	

	# simulations = 2	#nr of actions to simulate
	# output = torch.zeros(simulations, batch_size)-1	#dimensions are wrong, will be transposed later
	# simulated = []	#keeps track of all the action that have been simulated
	# unique_simulations = 0	#counts all unique simulations

	# for _ in range(simulations):
	# 	#randomly sample one char based on the distribution from the dataset
	# 	action = np.random.choice(np.arange(len(alphabet)), p = distribution)
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
	
	output = torch.zeros(len(alphabet), batch_size)	#dimensions are wrong, will be transposed later
	for action in range(len(alphabet)):
		output[action] = predMaxReward(expandTree(state, action), length, discriminator)

	output = output.transpose(0, 1)
	
	return output#simulated, output[:,:unique_simulations]

def fetch_sample(dataset_path):
	with open(dataset_path, "r") as source:
		for line in source:
			yield encode(["hallo"])


def encode(lines):
	assert isinstance(lines, list)	#input is a list of lines
	assert all(len(element) == len(lines[0]) for element in lines)	#all lines need to have equal length
	batch_size = len(lines)

	result = torch.zeros(len(lines[0]), batch_size, len(alphabet))
	for line_ix, line in enumerate(lines):
		for char_ix, char in enumerate(line):
			result[char_ix, line_ix, char_to_ix[char]] = 1

	return result

def decode(input):
	batch_size = len(input)
	result = []

	for batch_ix in range(batch_size):
		result.append("".join([ix_to_char[torch.argmax(vector).item()] for vector in input[batch_ix]]))
	return result
	
def rolloutPartialSequence(input, length):
	"""takes a incomplete sequence and appends n random actions"""
	output = torch.empty(length, input.shape[1], input.shape[2])
	output[:input.shape[0]] = input

	#randomly fill in the remaining values(rollout)
	for index in range(input.shape[0], length):
		batch = torch.zeros(input.shape[1], len(alphabet))
		for b in range(input.shape[1]):
			action = np.random.choice(np.arange(len(alphabet)), p = distribution)
			batch[b] = torch.zeros(len(alphabet))
			batch[b][action] = 1

		output[index] = batch
	return output

def roundOutput(input):
	output = torch.zeros(input.shape)
	for batch_ix, batch in enumerate(input):
		output[batch_ix, torch.argmax(batch)] = 1
	return output

def NLLLoss(input, target, use_baseline = False):
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
	
	#actually calculate the loss
	batch_size = input.shape[0]
	result = torch.empty(batch_size)
	for batch_ix in range(batch_size):
		result[batch_ix] = -1 * torch.dot(input[batch_ix], target[batch_ix])
		
	return torch.mean(result)