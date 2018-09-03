'''
Here we will load the initial trained embeddings model from gensim
and extract out the vectors from the vocab for each word, 
so that we can use these embeddings as an initialization to triplet loss 
embeddings.
We will also use the vocab
'''
import gensim.models.word2vec as word2vec
from gensim.models import KeyedVectors
import numpy as np
import pickle


MODEL_PATH = 'trained_word2vec_huge_vocab/dhcl_word2vec_model_300'
EMBEDDINGS_SAVE_PATH = 'trained_word2vec_huge_vocab/dhcl_word2vec_model_300_embeddings'
DICTI_DUMP_PATH = 'trained_word2vec_huge_vocab/dhcl_word2vec_model_300_vocabs_dicti_dump.dicti'

#load model
model = word2vec.Word2Vec.load(MODEL_PATH)

# loading initial embeddings
#initialEmbeddings = np.load(EMBEDDINGS_SAVE_PATH)

# we will create own own wordIndex vocab which will keep
# track in what order are the vectors getting stored
wordIndexVocab = {}

vectors = []
for word in model.wv.vocab:
	wordIndexVocab[word] = len(wordIndexVocab)
	vectors.append(model[word])

indexWordVocab = dict((v, k) for k, v in wordIndexVocab.items())


print( 'word index vocab length: ', len(wordIndexVocab))
print(wordIndexVocab)
print('index word vocab length: ', len(indexWordVocab))
print(indexWordVocab)
vectors = np.array(vectors)
print('shape of the vectors embeddings: ', vectors.shape)

np.save(EMBEDDINGS_SAVE_PATH, vectors)

print('saved the embeddings')

with open(DICTI_DUMP_PATH, 'wb') as f:
	pickle.dump([wordIndexVocab, indexWordVocab], f)
print('both dicti saved')
