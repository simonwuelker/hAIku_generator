# Since the outputs from the generator are not decoded and re-encoded when being fed to the discriminator,
# the dataset needs to feed the discriminator with encoded sentences. Because of this, a pretrained embedding layer is
# needed
# This code is pretty much just taken from 
# https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html
import logging
import gensim.downloader as api  # in case you want to use text8
from gensim.models.word2vec import Word2Vec

class Corpus:
	def __init__(self, data_path):
		self.data_path = data_path

	def __iter__(self):
		with open(self.data_path, "r", errors="ignore") as infile:
			for line in infile.read().splitlines():
				yield line.split()


# Output during training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# use text8 corpus as training data
# training_data = api.load('text8')

# or use the existing dataset(smaller but somewhat limited amount of words)
training_data = Corpus("../data/dataset_clean.txt")

# create and train model
model = Word2Vec(training_data, min_count=1, size=50)  # raise min_count if using a larger dataset

# save the model
model.save("../models/word2vec.model")
