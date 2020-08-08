import torch
from ordered_set import OrderedSet  # OrderedSet ensures that the word-token mappings will always be the same
import gensim


class Dataset(torch.utils.data.Dataset):
	def __init__(self, path):
		self.data, self.unique_tokens = self.loadData(path)

		self.word_to_ix = {word: ix for ix, word in enumerate(self.unique_tokens)}
		self.ix_to_word = {ix: word for ix, word in enumerate(self.unique_tokens)}
		self.model = gensim.models.KeyedVectors.load("models/word2vec.model")
		self.embedding_dim = 100  # this is the default value for gensim models

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		"""Returns a sample of shape [N, Seq_length, 1] at a certain index"""
		return self.encode(self.data[index])

	def loadData(self, path):
		with open(path, "r", encoding="utf8", errors="ignore") as infile:
			haikus = infile.read().splitlines()

		return haikus, OrderedSet([word for haiku in haikus for word in haiku.split()])

	def encode(self, haiku):
		""" Encodes a single line of text into either indices or word tensors"""
		words = haiku.split()
		result = torch.empty(len(words), self.embedding_dim)
		for index, word in enumerate(words):
			result[index] = torch.tensor(self.model.wv[word])
		
		return result

	def decode(self, tensor):
		""" Decodes either a """
		batch_size = tensor.shape[0]
		seq_length = tensor.shape[1]

		result = []
		for batch_ix in range(batch_size):
			batchstring = ""
			for word_vector in tensor[batch_ix]:
				batchstring += self.model.wv.most_similar(positive=[word_vector.numpy()])[0][0] + " "
			result.append(batchstring)

		return result
