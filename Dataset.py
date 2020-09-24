import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np


class Embedding:
	"""
	Combine all the functions around word embedddings. Similar to a gensim model instance
	which is not used here since gensim is annoying to use and takes a while to import.
	"""
	def __init__(self, gensim_model):
		self.embedding_dim = gensim_model.vector_size
		self.vocab = list(gensim_model.wv.vocab.keys())  # could just use word_to_ix.keys() to save memory
		self.word_to_ix = {word:gensim_model.wv.vocab[word].index for index, word in enumerate(self.vocab)}
		self.ix_to_word = {gensim_model.wv.vocab[word].index:word for index, word in enumerate(self.vocab)}

		# copy over word vectors from gensim
		self.word_vectors = torch.FloatTensor(gensim_model.wv.vectors)

		self.append_token("<n>", torch.zeros(self.embedding_dim) - 1) # torch.full doesnt support 1D Tensors
		self.append_token("<unk>", torch.ones(self.embedding_dim))
		self.append_token("<eos>", torch.zeros(self.embedding_dim))


	def append_token(self, token, vector):
		"""
		Add a single token with a given word vector to the model vocabulary

		Parameters:
				token(String): The token to add to the vocabulary
				vector(Tensor): The future corresponding vector for the token. Shape [embedding_dim]
		"""
		# update dictionaries and vocab
		self.vocab.append(token)
		self.word_to_ix[token] = len(self.vocab) - 1
		self.ix_to_word[len(self.vocab) - 1] = token

		# update word vectors
		new_word_vectors = torch.zeros(len(self.vocab), self.embedding_dim)
		new_word_vectors[:-1] = self.word_vectors  # copy over old vectors
		new_word_vectors[-1] = vector
		self.word_vectors = new_word_vectors

	def __getitem__(self, word):
		"""
		Turn a given word into vector. If the word is unknown, the <unk> token
		is used instead.
		Parameters:
				word(String): The word to convert into a vector

		Returns:
				word_vector(Tensor): The vector corresponding to the word. Shape [1, embedding_dim]
		"""
		try:
			index = self.word_to_ix[word]
		except KeyError:
			print("unable to find word: ", word)
			index = self.word_to_ix["<unk>"]

		word_vector = self.word_vectors[index].view(1, self.embedding_dim)
		return word_vector


	def most_similar(self, target_vector, n=5, single=False):
		"""
		Find the n most similar words to the target vector and return them as a list with 
		their corresponding distances.

		Parameters:
				target_vector(Tensor): The Tensor whose most similar vector is searched for
				n(int): The number of vectors to return, sorted by similarity, descending
				single(bool): whether or not to return only a single vector without additional data.
							  If set to True, the Parameter n will not be accounted for.

		Returns:
				if single is set to True:
					closest(String): The word with the Vector most similar to the target Vector

				if single is set to False:
					closest(zip): A zip object containing the closest words with their corresponding distances,
						   sorted by distance, descending
		
		"""
		distances = torch.zeros(len(self.vocab))
		dist = nn.PairwiseDistance(p=2)

		# get the distance for every single word in the vocab
		for index, vector in enumerate(self.word_vectors):
			distances[index] = dist(vector.view(1, -1), target_vector.view(1, -1))
		
		if single:
			closest = self.ix_to_word[torch.argmin(distances).item()]
			return closest
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
	def __init__(self, path_data, path_model, train_test=0.8):
		# load all the haikus from a file
		with open(path_data, "r", encoding="utf8", errors="ignore") as infile:
			self.data = infile.read().splitlines()

		self.embedding = torch.load(f"{path_model}/word2vec.model")

		self.train_test = train_test
		self.train_cap = int(len(self.data) * self.train_test)
		self.test_cap = len(self.data)

	def DataLoader(self, batch_size=1):
		"""
		Yield encoded Haikus as PackedSequences until one epoch has passed.

		Parameters:
				batch_size(int): The number of haikus to be returned per iteration

		Returns:
				packed_data(PackedSequence): the haikus with their corresponding lengths
		"""
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
		Encode a single line of text into a Pytorch Tensor.
		Unknown words are replaced with the <unk> token.
		
		Parameters:
				haiku(String): The haiku to be encoded

		Returns:
				result(Tensor): the resulting Tensor of Shape [sequence, embedding_dim]
		"""
		words = haiku.split()
		result = torch.empty(len(words), self.embedding.embedding_dim)
		for index, word in enumerate(words):
			result[index] = self.embedding[word]
		
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
				haiku += self.embedding.most_similar(word_vector, single=True) + " "
				
			haikus.append(haiku)

		return haikus
