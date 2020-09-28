import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np
import gensim


class Dataset(torch.utils.data.Dataset):
	def __init__(self, args):
		# load all the haikus from a file
		with open(args.data_path, "r", encoding="utf8", errors="ignore") as infile:
			self.data = infile.read().splitlines()

		self.word2vec = gensim.models.KeyedVectors.load(f"{args.model_path}/word2vec.model")

	def __len__(self):
		return len(self.data)

	def DataLoader(self, end, start=0, batch_size=1):
		"""
		Yield encoded Haikus as PackedSequences until one epoch has passed.

		Parameters:
				start(int): The index to start yielding samples from
				end(int): The index to end the epoch on
				batch_size(int): The number of haikus to be returned per iteration

		Returns:
				packed_data(PackedSequence): the haikus with their corresponding lengths
		"""
		for index in range(start, end, batch_size):
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
		Encode a single line of text into a Pytorch Tensor.
		Unknown words are replaced with the <unk> token.
		
		Parameters:
				haiku(String): The haiku to be encoded

		Returns:
				result(Tensor): the resulting Tensor of Shape [sequence, embedding_dim]
		"""
		words = haiku.split()
		result = torch.empty(len(words), self.word2vec.vector_size)
		for index, word in enumerate(words):
			try:
				result[index] = torch.from_numpy(self.word2vec[word])
			except KeyError:
				result[index] = torch.from_numpy(self.word2vec["<unk>"])
		
		return result

	def decode(self, haiku_enc):
		"""
		Decode the input into a list of Haikus

		Parameters:
				haiku_enc(Tensor, PackedSequence): The encoded Haiku of Shape [batch, sequence, embedding_dim]

		Returns:
				haikus (list): 1D List of Strings
		"""
		if isinstance(haiku_enc, PackedSequence):
			haiku_enc, lengths = pad_packed_sequence(haiku_enc, batch_first=True)
		else:
			# every haiku gets max length
			lengths = [haiku_enc.shape[1]] * haiku_enc.shape[0]

		batch_size = haiku_enc.shape[0]

		haikus = []
		for batch_ix in range(batch_size):
			haiku = ""
			for seq_ix in range(lengths[batch_ix]):
				word_vector = haiku_enc[batch_ix, seq_ix]
				haiku += self.word2vec.most_similar(positive=[word_vector.numpy()])[0][0] + " "
				
			haikus.append(haiku)

		return haikus
