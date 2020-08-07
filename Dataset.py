import torch
from ordered_set import OrderedSet  # OrderedSet ensures that the word-token mappings will always be the same


class Dataset(torch.utils.data.Dataset):
	def __init__(self, path):
		self.path = path
		self.data, self.unique_tokens = self.loadData()

		self.word_to_ix = {word: ix for ix, word in enumerate(self.unique_tokens)}
		self.ix_to_word = {ix: word for ix, word in enumerate(self.unique_tokens)}

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		"""Returns a sample of shape [N, Seq_length, 1] at a certain index"""
		return self.encode(self.data[index])

	def loadData(self):
		with open(self.path, "r", encoding="utf8", errors="ignore") as infile:
			haikus = infile.read().splitlines()

		return haikus, OrderedSet([word for haiku in haikus for word in haiku.split()])

	def encode(self, haiku, indices=True):
		""" Encodes a single line of text into either indices or word tensors"""
		words = haiku.split()

		if indices:
			result = torch.empty(len(words))
			for word_ix, word in enumerate(words):
				result[word_ix] = self.word_to_ix[word]
		else:
			result = torch.empty(len(words), self.embedding.embedding_dim)
			for word_ix, word in 
		return result

	def decode(self, tensor, indices=True):
		""" Decodes either a """
		#WIE KANN DAS NE GUTE IDEE SEINMIT DEN EMBEDDINGS...
		batch_size = tensor.shape[0]
		seq_length = tensor.shape[1]

		result = []
		for batch_ix in range(batch_size):
			batchstring = ""
			for seq_ix in range(seq_length):
				batchstring += self.ix_to_word[tensor[batch_ix, seq_ix].item()] + " "
			result.append(batchstring)

		return result
