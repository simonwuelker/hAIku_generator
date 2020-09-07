# Since the outputs from the generator are not decoded and re-encoded when being fed to the discriminator,
# the dataset needs to feed the discriminator with encoded sentences. Because of this, a pretrained embedding layer is
# needed
# This code is pretty much just taken from 
# https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html
# import logging
# import gensim.downloader as api  # in case you want to use text8
# from gensim.models.word2vec import Word2Vec

# class Corpus:
# 	def __init__(self, data_path):
# 		self.data_path = data_path

# 	def __iter__(self):
# 		with open(self.data_path, "r", errors="ignore") as infile:
# 			for line in infile.read().splitlines():
# 				yield line.split()


# # Output during training
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # use text8 corpus as training data
# training_data = api.load('text8')

# # or use the existing dataset(smaller but somewhat limited amount of words, wv may become meaningless)
# # training_data = Corpus("../data/dataset_clean.txt")

# # create and train model
# model = Word2Vec(training_data, min_count=5, size=80)  # raise min_count & size if using a larger dataset

# # save the model
# model.save("../models/word2vec.model")

# # run some tests on the model
# print(model.wv.most_similar(positive=["rain"]))
# print(model.wv.most_similar(positive=["man"]))
# print(model.wv.most_similar(positive=["tree"]))

import torch
import torch.nn as nn
import gensim
import pickle

model = gensim.models.KeyedVectors.load("../models/word2vec.model")

# because the gensim model kinda sucks im creating my own class that i can just pickle dump and reload
# also gensim doesnt support item assignment
class Word2Vec:
	def __init__(self, gensim_model):
		self.embedding_dim = gensim_model.vector_size
		self.vocab = list(gensim_model.wv.vocab.keys()) + ["<unk>", "<eos>"]

		self.word_to_ix = {word:index for index, word in enumerate(self.vocab)}
		self.ix_to_word = {index:word for index, word in enumerate(self.vocab)}

		# somehow append <eos> and <unk> to the end of wv
		self.word_vectors = torch.zeros(len(self.vocab), self.embedding_dim)
		self.word_vectors[:-2] = torch.FloatTensor(gensim_model.wv.vectors)  # copy over wv from gensim
		self.word_vectors[-2] = torch.ones(self.embedding_dim)  # <unk>
		self.word_vectors[-1] = torch.zeros(self.embedding_dim)  # <eos>

		# create the pytorch embedding layer with the 2 new tokens
		self.embedding = nn.Embedding.from_pretrained(self.word_vectors)


	def __getitem__(self, word):
		try:
			index = self.word_to_ix[word]
		except KeyError:
			print("unable to find word: ", word)
			index = self.word_to_ix["<unk>"]

		return self.embedding(torch.LongTensor([index]))


	def most_similar(self, target_vector, n=5, use_cosine=False):
		"""
		Finds the n most similar words to the target vector and returns them as a list.
		Can use either cosine distance or pairwise distance.
		"""
		similarities = torch.zeros(len(self.vocab))
		cos = nn.CosineSimilarity(dim=0, eps=1e-6)
		dist = nn.PairwiseDistance(p=2)

		for index, vector in enumerate(self.word_vectors):
			if use_cosine:
				similarity = cos(vector, target_vector)
			else:
				similarity = dist(vector, target_vector)
			similarities[index] = similarity

		# retrieve the n highest indices
		n_highest = torch.argsort(similarities, descending=True)[:n]

		words = []
		for index in n_highest:
			words.append(self.vocab[index])

		return self.vocab[torch.argmax(similarities)]

word2vec_model = Word2Vec(model)
print(word2vec_model["tree"])
print(word2vec_model["<eos>"])
print(word2vec_model["<unk>"])
print(word2vec_model.most_similar(word2vec_model["tree"], n=10)) # "so"
with open("../models/word2vec_.model", "w") as outfile:
	pickle.dump(word2vec_model, outfile)
