'''

'''
import pandas as pd
import numpy as np
import os
import pickle
import multiprocessing
import gensim.models.word2vec as word2vec


SEED = 2
SKILL_LIST_PATH = 'triplet_loss_embedding_skills.pickle'

# loading out skill
#skillsArray = np.load(SKILL_LIST_PATH)
with open(SKILL_LIST_PATH, 'rb') as f:
	skillsArray = pickle.load(f)
print(skillsArray[:200])

# now we will build a word2vec model in built from gensim

dimensions = 200
# here dienssions is the dimension of the vector that we want. 
#The more the dimensions
#the more accurate the model is, more general
#but computational cost will increase

minWordCount = 1
#this is the minimum threshold that a word should cross to get registered

#workers = multiprocessing.cpu_count()
# this is to use multiprocessing

contextSize = 7
# this is the length of the sentence that would b considered in one context

#downsample = 1e-5
#for frequent word we use downsampling
print('building wor2vec from gensim')
model = word2vec.Word2Vec(sg=1, seed=1, size = dimensions, min_count=minWordCount,
                              window=contextSize)
#here we have just made the model, not feed data into it

# now we will build vocabulary to model and then print how many word got there in vocab
#all wont come due to mincount parameteres and all

model.build_vocab(skillsArray)
print('Done with building vocab for model.')
print('vocab that got into length : ',model.wv.vocab.__len__())
#print('vocab :', model.wv.vocab)
# now we ll train the model with tokens

print('going into training. May take time...')

model.train(skillsArray, total_examples= model.corpus_count, epochs=model.iter)

print('Successfully done training !')
print('Now moving to saving model in the name trained')

if not os._exists('trained_word2vec'):
    os.makedirs('trained_word2vec')

model.save(os.path.join('trained_word2vec', 'model.w2v'))

print('Saved the trained model')
