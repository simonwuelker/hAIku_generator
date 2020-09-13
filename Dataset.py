import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
import gensim


class Embedding:
	def __init__(self, gensim_model):
		self.embedding_dim = gensim_model.vector_size
		self.vocab = list(gensim_model.wv.vocab.keys())  # could just use word_to_ix.keys() to save memory
		print("gensim vocab has ", len(self.vocab), " words")
		self.word_to_ix = {word:gensim_model.wv.vocab[word].index for index, word in enumerate(self.vocab)}
		self.ix_to_word = {gensim_model.wv.vocab[word].index:word for index, word in enumerate(self.vocab)}

		# append <eos> and <unk> to the end of wv
		self.vocab += ["<unk>", "<eos>"]

		# update dictionaries
		self.word_to_ix["<unk>"] = len(self.vocab) - 2
		self.word_to_ix["<eos>"] = len(self.vocab) - 1
		self.ix_to_word[len(self.vocab) - 2] = "<unk>"
		self.ix_to_word[len(self.vocab) - 1] = "<eos>"

		# create word vectors
		self.word_vectors = torch.zeros(len(self.vocab), self.embedding_dim)
		self.word_vectors[:-2] = torch.FloatTensor(gensim_model.wv.vectors)  # copy over wv from gensim
		self.word_vectors[-2] = torch.ones(self.embedding_dim)  # <unk>
		self.word_vectors[-1] = torch.zeros(self.embedding_dim)  # <eos>


	def __getitem__(self, word):
		try:
			index = self.word_to_ix[word]
		except KeyError:
			print("unable to find word: ", word)
			index = self.word_to_ix["<unk>"]

		return self.word_vectors[index].view(1, self.embedding_dim)


	def most_similar(self, target_vector, n=5, single=False):
		"""
		Finds the n most similar words to the target vector and returns them as a list with 
		their corresponding distances.
		If single is set to True, only the highest word and nothing else is returned.
		"""
		distances = torch.zeros(len(self.vocab))
		dist = nn.PairwiseDistance(p=2)

		# get the distance for every single word in the vocab
		for index, vector in enumerate(self.word_vectors):
			distances[index] = dist(vector.view(1, -1), target_vector.view(1, -1))
		
		if single:
			return self.ix_to_word[torch.argmin(distances).item()]
		else:
			# retrieve the n lowest indices
			n_highest = torch.argsort(distances)[:n]

			words = []
			lowest_distances = []
			for index in n_highest:
				words.append(self.ix_to_word[index.item()])
				lowest_distances.append(distances[index])

			return zip(words, lowest_distances)

class Dataset(torch.utils.data.Dataset):
	def __init__(self, path_data, path_model="models/word2vec.model", train_test=0.8):
		# load all the haikus from a file
		with open(path_data, "r", encoding="utf8", errors="ignore") as infile:
			self.data = infile.read().splitlines()

		self.embedding = torch.load(path_model)

		self.train_test = train_test
		self.train_cap = int(len(self.data) * self.train_test)
		self.test_cap = len(self.data)

	def DataLoader(self, end, start=0, batch_size=1):
		for index in range(start, self.train_cap, batch_size):
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

	def encode(self, haiku):
		"""
		Encodes a single line of text into a Tensor of Shape [num_words, embedding_dim].
		Sequence Packing is done later in the DataLoader Function.
		"""
		words = haiku.split()
		result = torch.empty(len(words), self.embedding.embedding_dim)
		for index, word in enumerate(words):
			result[index] = self.embedding[word]
		
		return result

	def decode(self, tensor):
		# this function is pretty slow
		batch_size = tensor.shape[0]
		seq_length = tensor.shape[1]

		haikus = []
		for batch_ix in range(batch_size):
			haiku = ""
			for word_vector in tensor[batch_ix]:
				haiku += self.embedding.most_similar(word_vector, single=True) + " "
			haikus.append(haiku)

		return haikus
