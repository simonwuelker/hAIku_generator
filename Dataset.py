import torch
from collections import Counter

class Dataset(torch.utils.data.Dataset):
	def __init__(self, path = "dataset.txt"):

		self.path = path
		self.data, self.unique_tokens = self.loadData()
	
		self.word_to_ix = {word:ix for ix, word in enumerate(self.unique_tokens)}
		self.ix_to_word = {ix:word for ix, word in enumerate(self.unique_tokens)}

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		input = " ".join(self.data[index].split()[:-1])	#remove last word from haiku
		target = " ".join(self.data[index].split()[1:])	#remove first word from target
		return self.encode([input]).view(-1, 1, 1), self.encode([target]).view(-1, 1, 1)

	def loadData(self):
		with open(self.path, "r", encoding="utf8", errors="ignore") as infile:
			haikus = infile.read()
		return haikus.split("\n"), self.get_unique_tokens(haikus.split())

	def get_unique_tokens(self, words, sort = True):
		if sort:
			word_counts = Counter(words)
			return sorted(word_counts, key=word_counts.get, reverse=True)

		return set(words)	#set saves memory in comparison to list

	def encode(self, context):
		context = [haiku.split() for haiku in context]
		result = torch.empty(len(context), len(context[0]))
		for batch_ix, batch in enumerate(context):
			for word_ix, word in enumerate(batch):
				result[batch_ix, word_ix] = self.word_to_ix[word]
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