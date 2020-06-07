import gensim	#gensim takes forever to import
from smart_open import open	#overwrite native open() function
import torch

class Dataloader(object):
    def __init__(self, path, ):
        self.filepath = path

    def __iter__(self):
        for line in open(self.filepath):
            yield line.split()

load_models = True

if load_models:
	#load model
	model = gensim.models.Word2Vec.load('models/word2vec.model')
else:
	#train model
	model = gensim.models.Word2Vec(Dataloader("dataset.txt"), min_count=1, size = 80)
	
model.save('models/word2vec.model')
