import torch
import numpy as np
import gensim

model = gensim.models.Word2Vec.load('models/word2vec.model')

def predMaxReward(state, length, discriminator):
	#predMaxReward(state) judges the quality of the given state [0,1] by performing n rollouts
	batch_size = state.shape[1]
	simulations = 50	#increase simulations to increase reward accuracy
	maxScores = torch.tensor([float("-inf")]*batch_size)
	for roll_ix in range(simulations):
		completed = rolloutPartialSequence(state, length)	#complete sequence to allow the discriminator to judge it
		scores = discriminator(completed)
		#print(f"{decode(completed)} score {scores[-1]}")
		#scores = scores[-1,:]
		for batch_ix in range(batch_size):
			if scores[batch_ix] > maxScores[batch_ix]:
				maxScores[batch_ix] = scores[batch_ix]
	return maxScores

def expandState(state, action):
	#expandState is the state transition function δ
	output = torch.zeros(state.shape[0]+1, state.shape[1], state.shape[2])
	output[:state.shape[0]] = state
	output[-1, :] = action
	return output

def bestAction(state, length, discriminator):
	batch_size = state.shape[1]
	bests = [[None, float("-inf")] for b in range(batch_size)]

	new = torch.empty(state.shape[0]+1, state.shape[1], state.shape[2])
	new[:-1] = state

	#action space is continuous therefore only simulate some actions
	for _ in range(25):
		action = torch.rand(model.vector_size)*3
		new[-1] = action
		#print(f"now simulating {model.wv.most_similar(positive = [action.detach().numpy()])} and thats nr {_}")
		reward = predMaxReward(new, length, discriminator)
		for batch_ix in range(batch_size):
			if bests[batch_ix][1] < reward[batch_ix]:
				bests = [action, reward[batch_ix]]
	#print(f" best word was {model.wv.most_similar(positive = [bests[0].numpy()])}")
	return bests

def fetch_sample(dataset_path):
	with open(dataset_path, "r") as source:
		for line in source:
			yield encode(line)[:2]


def encode(line):
	result = torch.zeros(len(line.split()), 1, model.vector_size)
	for index, word in enumerate(line.split()):
		try:
			result[index, 0] = torch.from_numpy(model[word])
		except KeyError:
			print(f"Unknown word {word}")

	return result

def decode(tensor):
	assert tensor.shape[1] == 1	#can currently only handle one batch at a time

	result = "".join(model.wv.most_similar(positive = [element[0].detach().numpy()])[0][0] + " " for element in tensor)
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
		batch = torch.zeros(input.shape[1], model.vector_size)
		for b in range(input.shape[1]):
			batch[b] = torch.rand(model.vector_size)*3

		output[index] = batch

	#print(f"after rollout: {decode(output)}")
	return output 	#round the outputs to the nearest character

def roundOutput(input):
	output = torch.empty(input.shape)
	for time_ix, timestep in enumerate(input):
		for batch_ix, batch in enumerate(timestep):
			output[time_ix, batch_ix] = torch.from_numpy(model.wv[model.wv.most_similar(positive = [batch.detach().numpy()])[0][0]])
	return output