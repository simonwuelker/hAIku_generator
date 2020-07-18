import torch
import torch.utils.data
import string


class Dataset(torch.utils.data.Dataset):

	def __init__(self, path):
		self.path = path
		self.data = self.loadData()
		self.unique_tokens = string.ascii_lowercase + ", "

		self.token_to_ix = {token: ix for ix, token in enumerate(self.unique_tokens)}
		self.ix_to_token = {ix: token for ix, token in enumerate(self.unique_tokens)}

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		input = self.data[index][:-1]  # remove last word from haiku
		target = self.data[index][1:]  # remove first word from target
		return self.encode([input]).view(-1, 1, 1), self.encode([target]).view(-1, 1, 1)

	def loadData(self):
		"""
		Loads all the Haikus from the input file
		and splits them into a list
		"""
		with open(self.path, "r", encoding="utf8", errors="ignore") as infile:
			haikus = infile.read()
		return haikus.split("\n")

	def encode(self, haikus):
		"""
		Encodes a given list of haikus
		into a 3d matrix
		All Haikus must have equal length
		"""

		result = torch.empty(len(haikus), len(haikus[0]))
		for index, haiku in enumerate(haikus):
			result[index] = torch.tensor([self.token_to_ix[token] for token in haiku])
		return result.transpose(1, 0)

	def decode(self, tensor):
		"""
		Decodes a tensor of shape[Sequence_Length, batch_size, num_Classes]
		into  a list of strings
		"""
		seq_length = tensor.shape[0]
		batch_size = tensor.shape[1]

		result = []
		for batch_ix in range(batch_size):
			batchstring = ""
			for seq_ix in range(seq_length):
				batchstring += self.ix_to_token[tensor[seq_ix, batch_ix].item()]
			result.append(batchstring)

		return result
