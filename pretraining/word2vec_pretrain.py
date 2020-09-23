# enable imports from parent directory
import sys
import os
sys.path.append(os.path.realpath(".."))

from Dataset import Embedding
import logging
import gensim.downloader as api  # in case you want to use text8
from gensim.models.word2vec import Word2Vec
from gensim.models import Phrases
import torch
import gensim#



# # Output during training
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # use text8 corpus as training data
# training_data = api.load('text8')

# # use the phrase model to recognize bigrams like "White House" or "Climate Change"
# bigram_transformer = Phrases(training_data)

# # create and train model
# model = Word2Vec(bigram_transformer[training_data], size=80)

model = gensim.models.KeyedVectors.load("../models/word2vec_GENSIM.model")

# save the word vectors
with open("../models/word2vec.model", "wb") as outfile:
	torch.save(Embedding(model), outfile)
