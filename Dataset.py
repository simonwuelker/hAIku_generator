# https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
import gensim

class Dataset(torch.utils.data.Dataset):
	def __init__(self, path_data, path_model="models/word2vec.model", train_test=0.8):
		# load all the haikus from a file
		with open(path_data, "r", encoding="utf8", errors="ignore") as infile:
			self.data = infile.read().splitlines()

		self.model = gensim.models.KeyedVectors.load(path_model)
		self.embedding_dim = self.model.vector_size  # default is 100

		self.train_test = train_test
		self.train_cap = int(len(self.data) * train_test)
		self.test_cap = len(self.data)

	def training_iterator(self, batch_size):
		for index in range(0, self.train_cap, batch_size):
			unpadded_data = []
			lengths = []  # lengths are needed for sequence packing
			for j in range(batch_size):
				haiku = self.data[index + j]
				lengths.append(len(haiku.split()))
				unpadded_data.append(self.encode(haiku))

			# padd the haikus and pack them into a PackedSequence Object
			padded_data = pad_sequence(unpadded_data, batch_first=True)
			packed_data = pack_padded_sequence(padded_data, lengths, batch_first=True, enforce_sorted=False)
			yield packed_data

	def testing_iterator(self, batch_size):
		for index in range(self.train_cap, self.test_cap, batch_size):
			yield self.encode(self.data[index])

	def encode(self, haiku):
		""" Encodes a single line of text into either indices or word tensors"""
		haiku += " <eos>"
		words = haiku.split()
		result = torch.empty(len(words), self.embedding_dim)
		for index, word in enumerate(words):
			result[index] = torch.tensor(np.copy(self.model.wv[word]))
		
		return result

	def decode(self, tensor):
		# this function is pretty slow
		tensor = tensor.detach()
		batch_size = tensor.shape[0]
		seq_length = tensor.shape[1]

		result = []
		for batch_ix in range(batch_size):
			batchstring = ""
			for word_vector in tensor[batch_ix]:
				batchstring += self.model.wv.most_similar(positive=[word_vector.numpy()])[0][0] + " "
			result.append(batchstring)

		return result

# MOVE TO NOTEBOOK LATER, CAP AT 15, min 12

# import torch.utils.data  # cant inherit from torch.utils.data.Dataset otherwise
# from collections import Counter
# import matplotlib.pyplot as plt
# d = Dataset("data/dataset_clean.txt")
# lengths = [len(sample.split()) for sample in d.data]
# counter = Counter(lengths)
# occurences = [counter[key] for key in counter.keys()]
# print(occurences)
# plt.bar(x=list(counter.keys()), height=occurences)
# plt.show()

# dataloader_train = torch.utils.data.DataLoader(d, batch_size=2)
# for element in dataloader_train:
# 	print(element, type(element), element.shape)


