import torch
import numpy as np
import string
import sys

alphabet = "halo"#string.ascii_lowercase + "," + " "
char_to_ix = {char:ix for ix, char in enumerate(alphabet)}
ix_to_char = {ix:char for ix, char in enumerate(alphabet)}

def predMaxReward(state, length, discriminator):
	#predMaxReward(state) judges the quality of the given state [0,1] by performing n rollouts
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
	#expandTree is the state transition function δ
	batch_size = state.shape[1]
	seq_length = state.shape[0]
	output = torch.zeros(seq_length+1, batch_size, len(alphabet))
	output[:seq_length] = state
	output[-1, :, action] = 1
	return output

def Q(state, length, discriminator):
	#state = state[1:]#des is falsch oder macht zumindest wenig sinn
	#Q(state) returns the quality of all possible actions that can be performed in the given state(assumes state has Markovian Property)
	batch_size = state.shape[1]
	output = torch.empty(len(alphabet), batch_size)#dimensions are wrong, will be transposed later
	softmax = torch.nn.LogSoftmax(dim = 1)

	for action in range(len(alphabet)):
		output[action] = predMaxReward(expandTree(state, action), length, discriminator)

	output = output.transpose(0, 1)

	return softmax(output)

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
	sys.stdout.flush()# Linux sometimes buffers output. bad. prevent.
	
def rolloutPartialSequence(input, length):
	output = torch.empty(length, input.shape[1], input.shape[2])

	output[:input.shape[0]] = input


	#randomly fill in the remaining values(rollout)
	for index in range(input.shape[0], length):
		batch = torch.zeros(input.shape[1], len(alphabet))
		for b in range(input.shape[1]):
			batch[b] = torch.zeros(len(alphabet))
			batch[b][np.random.randint(0, len(alphabet)-1)] = 1

		output[index] = batch
	return output

def roundOutput(input):
	output = torch.zeros(input.shape)
	for batch_ix, batch in enumerate(input):
		output[batch_ix, torch.argmax(batch)] = 1
	return output