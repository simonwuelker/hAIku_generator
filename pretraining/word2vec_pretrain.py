from Dataset import Embedding
import logging
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models import Phrases
import torch
import gensim

def train(args):
	# Output during training
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	# use text8 corpus as training data, haikus dont provide sufficient context
	training_data = api.load('text8')

	# use the phrase model to recognize bigrams like "White House" or "Climate Change"
	bigram_transformer = Phrases(training_data)

	# create and train model
	model = Word2Vec(bigram_transformer[training_data], size=args.embedding_dim)

	# save the word vectors
	embedding = Embedding(model)
	with open(f"{args.model_path}/word2vec.model", "wb") as outfile:
		torch.save(embedding, outfile)

	return embedding

