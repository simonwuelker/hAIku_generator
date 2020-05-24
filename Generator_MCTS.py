import numpy as np
import torch

import mido

import MidiTools
import pickle

#Single Player implementation of the Monte Carlo search tree algorithm for use in the generator model of SeqGan
#The node selection algorithm is UCT(Upper Confidence bound for trees)
class MCTSNode:
	#c is the exploration factor in the UCT algorithm
	c = 1.4
	p = np.load("distribution.npy")
	counter = 0
	allNodes = []
	branching_factor = 0
	def __init__(self, parent, state):
		self.parent = parent
		self.state = state

		self.children = np.zeros(MCTSNode.branching_factor, dtype = np.uint32)
		self.scores = np.zeros(MCTSNode.branching_factor)
		self.simulations = np.zeros(MCTSNode.branching_factor)

		self.last_move = None

		self.id = MCTSNode.counter
		MCTSNode.allNodes.append(self)
		MCTSNode.counter += 1

	def chooseMove(self, allow_exploration = True):
		if allow_exploration:
			#check for unexplored nodes
			for move_ix in range(MCTSNode.branching_factor):
				if self.simulations[move_ix] == 0:
					self.last_move = move_ix
					return move_ix, True

		UCTScores = np.zeros(MCTSNode.branching_factor)

		total_simulations = sum(self.simulations)

		#calculate the uct value of each node
		for move_ix in range(MCTSNode.branching_factor):
			#Add the avg score to each moves UCT score(this is the 'Exploitation' part)
			UCTScores[move_ix] += self.scores[move_ix]/self.simulations[move_ix]

			if allow_exploration:
				#Add the whole 'Exploration' part
				UCTScores[move_ix] += MCTSNode.c * np.sqrt(np.log(total_simulations)/self.simulations[move_ix])

		#return the move with the highest value
		move = np.argmax([0 if np.isnan(score) else score for score in UCTScores])
		self.last_move = move
		return move, False

	def backpropagate(self, score):
		self.simulations[self.last_move] += 1
		self.scores[self.last_move] += score

		#recursively backpropagate the score through the search tree(Source node has parent None)
		if self.parent != None:
			MCTSNode.allMCTSNodes[self.parent].backpropagate(score)

def expand(state, move):
	for index, element in enumerate(state):
		if element == -1:
			state[index] = move
			return state
	return "AHHH DAS ARRAY IST SCHON VOLLLLLL ALARM"

def cycle(start_node, maxMoves, allow_exploration = True):
	moves_left = maxMoves

	current_node = start_node
	moveTracker = []
	rollout = False

	#SELECTION
	while not rollout and moves_left > 0:
		move, rollout = current_node.chooseMove(allow_exploration = allow_exploration)
		moveTracker.append(move)
		moves_left -= 1
		if not rollout:
			try:
				current_node = MCTSNode.allMCTSNodes[current_node.children[move]]
			except:
				print(move, current_node.children)
		else:
			#EXPANSION
			previous_node = current_node
			current_node = MCTSNode(previous_node.id, expand(current_node.state.copy(), move))
			previous_node.children[move] = MCTSNode.counter-1

	if moves_left > 0:
		#SIMULATION
		state = current_node.state.copy()
		for x in range(moves_left):
			move = np.random.choice(np.arange(MCTSNode.branching_factor), p = MCTSNode.p)
			state = expand(state, move)
			moveTracker.append(move)

	return moveTracker, current_node.parent


class generator():
	def __init__(self, sequence_length, branching_factor, c = 1.4):
		#Monte Carlo
		MCTSNode.c = c
		MCTSNode.branching_factor = branching_factor
		self.start_node = MCTSNode(None, np.empty(sequence_length)-1)
		

		self.sequence_length = sequence_length
		self.branching_factor = branching_factor
		self.losses = []

		#last node is the node to start backpropagating from
		self.last_node = None

	def __call__(self, sequence_length, allow_exploration = True):
		sequence, self.last_node = cycle(self.start_node, sequence_length, allow_exploration = allow_exploration)
		return sequence

	def next(self, sequence_length, allow_exploration = True):
		sequence = self(sequence_length, allow_exploration)
		return MidiTools.OneHotEncode(torch.Tensor(sequence)).unsqueeze(1)

	def optimize(self, score):
		MCTSNode.allMCTSNodes[self.last_node].backpropagate(score)

	#die ganze MCTSNode klasse mit pickle zu speicher wäre besser
	def saveModel(self, path):
		with open(path, "wb") as out_file:
			pickle.dump(MCTSNode.allMCTSNodes, out_file)

	def loadModel(self, path):
		with open(path, "rb") as in_file:
			MCTSNode.allMCTSNodes = pickle.load(in_file)
		MCTSNode.counter = len(MCTSNode.allMCTSNodes)

	def save_example(self, length = 100, print_output = True):
		output = self(length, allow_exploration = True)
		mid = MidiTools.decode(output)

		if print_output:
			for msg in mid:
				print(msg)

		mid.save("output.mid")




