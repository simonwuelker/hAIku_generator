# Since the outputs from the generator are not decoded and re-encoded when being fed to the discriminator,
# the dataset needs to feed the discriminator with encoded sentences. Because of this, a pretrained embedding layer is
# needed
# This code is pretty much just taken from 
# https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html

# Output during training
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# get corpus
import gensim.downloader as api
corpus = api.load('text8')

# create and train model
from gensim.models.word2vec import Word2Vec
model = Word2Vec(corpus)

# save the model
model.save("../models/word2vec.model")
