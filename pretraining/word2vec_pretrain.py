import logging
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import torch
import gensim
import numpy as np

def train(args):
	# Output during training
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	# use text8 corpus as training data, haikus dont provide sufficient context
	training_data = api.load('text8')

	# use the phrase model to recognize bigrams like "White House" or "Climate Change"
	bigram_model = Phrases(training_data)
	# Export the trained model = use less RAM, faster processing. Model updates no longer possible.
	bigrams = Phraser(bigram_model)

	# # create and train model
	model = Word2Vec(bigrams[training_data], size=args.embedding_dim)

	word_list = list(model.wv.vocab.keys())
	vector_list = [model[word] for word in word_list]

	# the basic model doesnt seem to be supporting item assignment
	# but WordEmbeddingsKeyedVectors does
	kv = WordEmbeddingsKeyedVectors(args.embedding_dim)
	kv.add(word_list, vector_list)

	kv.add(["<eos>", "<n>", "<unk>"], np.random.rand(3, args.embedding_dim))

	# just to be safe, clear the cache of normalized vectors
	# as i had a similar issue as https://github.com/RaRe-Technologies/gensim/issues/2532
	del kv.vectors_norm

	# save the new models
	bigrams.save(f"{args.model_path}/bigram.model")
	kv.save(f"{args.model_path}/word2vec.model")
