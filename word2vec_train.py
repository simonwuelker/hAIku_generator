import gensim	#gensim takes forever to import
from smart_open import open	#overwrite native open() function
import torch

class Dataloader(object):
    def __init__(self, path, ):
        self.filepath = path

    def __iter__(self):
        for line in open(self.filepath):
            yield gensim.utils.simple_preprocess(line)	#generator saves RAM in comparison to one huge string

min_count = 1 #minimal number of times the word has to occur in the corpus
data = Dataloader("dataset.txt")

model = gensim.models.Word2Vec(data, min_count=min_count, size = 80)
dictionary = gensim.corpora.Dictionary(data)

remove_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < min_count]
dictionary.filter_tokens(remove_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed

dictionary.save('models/word2vec.dict')
model.save('models/word2vec.model')
