import torch
import numpy as np
import gensim

class Dataset(torch.utils.data.Dataset):
	def __init__(self, path_data, path_model="models/word2vec.model"):
		# load all the haikus from a file
		with open(path_data, "r", encoding="utf8", errors="ignore") as infile:
			self.data = infile.read().splitlines()

		self.model = gensim.models.KeyedVectors.load("models/word2vec.model")
		self.embedding_dim = self.model.vector_size  # default is 100

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		"""Returns a sample of shape [Seq_length, embedding dim] at a certain index"""
		return self.encode(self.data[index])

	def encode(self, haiku):
		""" Encodes a single line of text into either indices or word tensors"""
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
