import Generator_MCTS
import MidiTools
import os
import random
import numpy as np
import pickle

modelsave_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/BW-KI-2020/models/"
dataset_path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/BW-KI-2020/notewise/"
sample_size = 32
dataloader = MidiTools.fetch_sample(sample_size, dataset_path, encode=False)
load_search_tree = False
Generator_MCTS.MCTSNode.branching_factor = MidiTools.vocab_size

if load_search_tree:
	Generator_MCTS.generator.loadModel(modelsave_path + "Generator_MCTS.pt")
	start_node = Generator_MCTS.MCTSNode.allNodes[0]
else:
	start_node = Generator_MCTS.MCTSNode(None, np.empty(sample_size)-1)



for i in range(10000):
	print(f"Sample Nr.{i}")

	current_node = start_node
	sample = next(dataloader)

	for note in sample:
		current_node.simulations[note] += 1
		current_node.scores[note] += 1

		#test if this was the first time this node was simulated
		if current_node.simulations[note] == 1:
			previous_node = current_node
			current_node = Generator_MCTS.MCTSNode(previous_node.id, Generator_MCTS.expand(current_node.state.copy(), note))
			previous_node.children[note] = Generator_MCTS.MCTSNode.counter-1
		else:
			current_node = Generator_MCTS.MCTSNode.allNodes[current_node.children[note]]

with open(modelsave_path + "Generator_MCTS_pretrained.pt", "wb") as out_file:
	pickle.dump(Generator_MCTS.MCTSNode.allNodes, out_file)


#TEST
sequence = np.empty(sample_size)
current_node = start_node
for step in range(sample_size):
	print(f"optimal move: {np.argmax(current_node.scores)}")
	sequence[step] = np.argmax(current_node.scores)

	current_node = Generator_MCTS.MCTSNode.allNodes[current_node.children[np.argmax(current_node.scores)]]
MidiTools.playMidi(MidiTools.decode(sequence))