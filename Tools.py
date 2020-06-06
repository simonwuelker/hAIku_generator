import string
import torch
import numpy as np

alphabet = list(string.ascii_lowercase + " " + ",")

def predMaxReward(state, length, discriminator):
	#predMaxReward(state) judges the quality of the given state [0,1] by performing n rollouts
	batch_size = state.shape[1]
	simulations = 50	#increase simulations to increase reward accuracy
	maxScores = torch.tensor([float("-inf")]*batch_size)
	for roll_ix in range(simulations):
		completed = rolloutPartialSequence(state, length)	#complete sequence to allow the discriminator to judge it
		scores = discriminator(completed)
		scores = scores[-1,:]
		for batch_ix in range(batch_size):
			if scores[batch_ix] > maxScores[batch_ix]:
				maxScores[batch_ix] = scores[batch_ix]
	return maxScores

def expandState(state, action):
	#expandState is the state transition function δ
	output = torch.zeros(state.shape[0]+1, state.shape[1], state.shape[2])
	output[:state.shape[0]] = state
	output[-1, :, action] = 1
	return output

def Q(state, length, discriminator):
	#Q(state) returns the quality of all possible actions that can be performed in the given state(assumes state has Markovian Property)
	batch_size = state.shape[1]
	output = torch.empty(len(alphabet), batch_size)#dimensions are wrong, will be transposed later

	for action in range(len(alphabet)):
		output[action] = predMaxReward(expandState(state, action), length, discriminator)

	output = output.transpose(0, 1)
	return output

def fetch_sample(dataset_path):
	with open(dataset_path, "r") as source:
		for line in source:
			yield encode(line)

def removeInvalid(text):
	result = ""
	for char in text.lower():
		if char in alphabet:
			result += char
	return result

def encode(line):
	input = removeInvalid(line)
	result = torch.zeros(len(input), 1, len(alphabet))
	for index, char in enumerate(input):
		result[index, 0, alphabet.index(char)] = 1

	return result

def decode(tensor):
	assert tensor.shape[1] == 1	#can currently only handle one batch at a time
	result = ""
	for element in tensor:
		result += alphabet[torch.argmax(element[0])]

	return result

def progressBar(start_time, time_now, training_time, episode):
	bar_length = 20
	elapsed = time_now - start_time
	num_filled = int(round((elapsed/training_time)*bar_length, 0))
	result = "|"

	result += "█"*num_filled

	result += "―"*(bar_length-num_filled)

	result += f"| {round(((elapsed/training_time)*100), 1)}% - ({round(elapsed, 1)}s elapsed) - Episode Nr.{episode}"
	print(result)
	print()
	
def rolloutPartialSequence(input, length):
	output = torch.empty(length, input.shape[1], input.shape[2])

	output[:input.shape[0]] = input

	#randomly fill in the remaining values(rollout)
	for index in range(input.shape[0], length):
		batch = torch.zeros(input.shape[1], len(alphabet))
		for b in range(input.shape[1]):
			batch[b] = torch.zeros(len(alphabet))
			batch[b][np.random.randint(length)] = 1

		output[index] = batch
	return output