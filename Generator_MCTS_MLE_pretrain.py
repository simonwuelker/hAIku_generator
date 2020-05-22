import mcts
import Converter
import os
import random
import numpy as np

modelsave_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/BW-KI-2020/models/"
dataset_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/Datensatz/notewise/"
sample_size = 32
dataloader = Converter.fetch_sample(sample_size, dataset_path, encode=False)
load_search_tree = False
mcts.Node.branching_factor = Converter.vocab_size
if load_search_tree:
	mcts.loadTree(modelsave_path + "Generator_MCTS.pt")
	start_node = mcts.Node.allNodes[0]
else:
	start_node = mcts.Node(None, np.empty(sample_size)-1)

current_node = start_node

sample = next(dataloader)


for note in sample:
	current_node.simulations[note] += 1
	current_node.scores[note] += 1


	if current_node.simulations[note] == 0:
		previous_node = current_node
		current_node = mcts.Node(previous_node.id, mcts.expand(current_node.state.copy(), note))
		previous_node.children[move] = mcts.Node.counter-1
	else:
		current_node = mcts.Node.allNodes[current_node.children[note]]



mcts.saveTree(modelsave_path + "Generator_MCTS.pt")

#TEST
import Generator_MCTS

generator = Generator_MCTS.generator(sample_size, Converter.vocab_size)
generator.save_example()
