import mcts

import numpy as np
import torch

import mido

import Converter

class generator():
	def __init__(self, sequence_length, branching_factor, c = 1.4):
		#Monte Carlo
		mcts.Node.c = c
		mcts.Node.branching_factor = branching_factor
		self.start_node = mcts.Node(None, np.empty(sequence_length)-1)
		

		self.sequence_length = sequence_length
		self.branching_factor = branching_factor
		self.losses = []

		#last node is the node to start backpropagating from
		self.last_node = None

	def __call__(self, sequence_length, encode = True):
		sequence, self.last_node = mcts.cycle(self.start_node, sequence_length)
		if encode:
			return Converter.OneHotEncode(torch.Tensor(sequence)).unsqueeze(1)
		else:
			return sequence

	def optimize(self, score):
		mcts.Node.allNodes[self.last_node].backpropagate(score)

	def saveModel(self, path):
		mcts.saveTree(path)

	def loadModel(self, path):
		mcts.loadTree(path)

	def play_example(array):
		mid = Converter.toMidi(array)
		port = mido.open_output()

		for msg in mid.play():
			if print_msg:
				print(msg)
			port.send(msg)

	def save_example(self, length = 100):
		output = self(length, encode = False)
		mid = Converter.decode(output)
		mid.save("output.mid")

	def train(self):
		pass

	def eval(self):
		pass

