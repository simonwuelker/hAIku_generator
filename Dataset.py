import torch
import torch.utils.data
import torch.nn.functional as F
import string


class Dataset(torch.utils.data.Dataset):

	def __init__(self, path):
		self.path = path
		self.unique_tokens = string.ascii_lowercase + ", "

		self.token_to_ix = {token: ix for ix, token in enumerate(self.unique_tokens)}
		self.ix_to_token = {ix: token for ix, token in enumerate(self.unique_tokens)}

		# load the data from the input file
		with open(self.path, "r", encoding="utf8", errors="ignore") as infile:
			self.data = infile.read().split("\n")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		sample = self.encode(self.data[index])
		return sample

	def encode(self, haiku):
		"""encodes a single haiku"""
		result = F.one_hot(torch.tensor([self.token_to_ix[char] for char in haiku]).long(), len(self.unique_tokens)).float()
		return result

	def decode(self, tensor):
		"""
		Decodes a tensor of shape[batch_size, seq_length, num_Classes]
		into  a list of strings
		"""
		seq_length = tensor.shape[1]
		batch_size = tensor.shape[0]

		result = []
		for batch_ix in range(batch_size):
			batchstring = ""
			for seq_ix in range(seq_length):
				batchstring += self.ix_to_token[torch.argmax(tensor[batch_ix, seq_ix]).item()]
			result.append(batchstring)

		return result
