import torch


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

		return haikus, set([word for haiku in haikus for word in haiku.split()])

	def encode(self, haiku):
		""" Encodes a single line of text """
		words = haiku.split()
		result = torch.empty(len(words), 1)
		for word_ix, word in enumerate(words):
			result[word_ix] = self.word_to_ix[word]
		return result

	def decode(self, tensor):
		seq_length = tensor.shape[0]
		batch_size = tensor.shape[1]

		result = []
		for batch_ix in range(batch_size):
			batchstring = ""
			for seq_ix in range(seq_length):
				batchstring += self.ix_to_word[tensor[seq_ix, batch_ix].item()]
			result.append(batchstring)

		return result
