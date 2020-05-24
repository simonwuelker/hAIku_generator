#Single Player implementation of the Monte Carlo search tree algorithm for use in the generator model of SeqGan
#The node selection algorithm is UCT(Upper Confidence bound for trees)
import numpy as np
import pickle

class Node:
	#c is the exploration factor in the UCT algorithm
	c = 1.4

	counter = 0
	allNodes = []
	branching_factor = 0
	def __init__(self, parent, state):
		self.parent = parent
		self.state = state

		self.children = np.zeros(Node.branching_factor, dtype = np.uint32)
		self.scores = np.zeros(Node.branching_factor)
		self.simulations = np.zeros(Node.branching_factor)

		self.last_move = None

		self.id = Node.counter
		Node.allNodes.append(self)
		Node.counter += 1

	def chooseMove(self, allow_exploration = True):
		if allow_exploration:
			#check for unexplored nodes
			for move_ix in range(Node.branching_factor):
				if self.simulations[move_ix] == 0:
					self.last_move = move_ix
					return move_ix, True

		UCTScores = np.zeros(Node.branching_factor)

		total_simulations = sum(self.simulations)

		#calculate the uct value of each node
		for move_ix in range(Node.branching_factor):
			#Add the avg score to each moves UCT score(this is the 'Exploitation' part)
			UCTScores[move_ix] += self.scores[move_ix]/self.simulations[move_ix]

			if allow_exploration:
				#Add the whole 'Exploration' part
				UCTScores[move_ix] += Node.c * np.sqrt(np.log(total_simulations)/self.simulations[move_ix])

		#return the move with the highest value
		move = np.argmax([0 if np.isnan(score) else score for score in UCTScores])
		self.last_move = move
		return move, False

	def backpropagate(self, score):
		self.simulations[self.last_move] += 1
		self.scores[self.last_move] += score

		#recursively backpropagate the score through the search tree(Source node has parent None)
		if self.parent != None:
			Node.allNodes[self.parent].backpropagate(score)
#optimaler wäre es die ganze klasse mit pickle zu speichern
def saveTree(path):
	with open(path, "wb") as out_file:
		pickle.dump(Node.allNodes, out_file)

def loadTree(path):
	with open(path, "rb") as in_file:
		Node.allNodes = pickle.load(in_file)
	Node.counter = len(Node.allNodes)

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
			current_node = Node.allNodes[current_node.children[move]]
		else:
			#EXPANSION
			previous_node = current_node
			current_node = Node(previous_node.id, expand(current_node.state.copy(), move))
			previous_node.children[move] = Node.counter-1

	
	if moves_left > 0:
		#SIMULATION
		state = current_node.state.copy()
		for x in range(moves_left):
			move = np.random.randint(Node.branching_factor)
			state = expand(state, move)
			moveTracker.append(move)

	return moveTracker, current_node.parent