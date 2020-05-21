#Single Player implementation of the Monte Carlo search tree algorithm for use in the generator model of SeqGan
#The node selection algorithm is UCT(Upper Confidence bound for trees)
import numpy as np
import pickle

class Node:
	#c is the exploration factor in the UCT algorithm
	c = 1.4

	counter = 0
	allNodes = []
	def __init__(self, parent, state, branching_factor):
		self.parent = parent
		self.state = state
		self.branching_factor = branching_factor

		self.children = np.zeros(branching_factor, dtype = np.uint32)
		self.scores = np.zeros(branching_factor)
		self.simulations = np.zeros(branching_factor)

		self.last_move = None

		allNodes.append(self)
		Node.counter += 1

	def chooseMove(self):
		#check for unexplored nodes
		for move_ix in range(self.branching_factor):
			if self.simulations[move_ix] == 0:
				self.last_move = move_ix
				return move_ix, True

		UCTScores = np.zeros(self.branching_factor)#[0 for _ in range(self.branching_factor)]

		total_simulations = sum(self.simulations)

		#calculate the uct value of each node
		for move_ix in range(self.branching_factor):
			#Add the avg score to each moves UCT score(this is the 'Exploitation' part)
			UCTScores[move_ix] += self.scores[move_ix]/self.simulations[move_ix]
			#Add the whole 'Exploration' part
			UCTScores[move_ix] += Node.c * np.sqrt(np.log(total_simulations)/self.simulations[move_ix])

		#return the move with the highest value
		move = np.argmax(UCTScores)
		self.last_move = move
		return move, False

	def backpropagate(self, score):
		self.simulations[self.last_move] += 1
		self.scores[self.last_move] += score

		#recursively backpropagate the score through the search tree(Source node has parent None)
		if self.parent != None:
			self.parent.backpropagate(score)

	def saveTree(path):
		with open(path, "wb") as out_file:
			pickle.dump(Node.allNodes, out_file)

	def loadTree(path):
		with open(path, "rb") as in_file:
			Node.allNodes = pickle.load(in_file)

def cycle(start_node):
	moves_left = maxMoves

	current_node = start_node
	moveTracker = []
	rollout = False
	win = False

	#SELECTION
	while not rollout and not win and moves_left > 0:
		move, rollout = current_node.chooseMove()
		moveTracker.append(move)
		moves_left -= 1
		if not rollout:
			current_node = Node.allNodes[current_node.children[move]]
		else:
			#EXPANSION
			previous_node = current_node
			current_node = Node(previous_node, Game.expand(current_node.state.copy(), move))
			previous_node.children[move] = Node.counter-1

		win = Game.testWin(current_node.state)
	
	if not win and moves_left > 0:
		#SIMULATION
		state = current_node.state.copy()
		for x in range(moves_left):
			state = Game.expand(state, np.random.randint(9))
			if Game.testWin(state):
				win = True
				break
	


	#BACKPROPAGATION
	print("Win:", win)
	current_node.parent.backpropagate(win)
	print(moveTracker)
	return win



class Environment:
	def testWin(self, state):
		pass

	def expand(self, state, move):
		pass