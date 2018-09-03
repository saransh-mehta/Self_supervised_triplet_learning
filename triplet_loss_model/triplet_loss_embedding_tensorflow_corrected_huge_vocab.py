'''
In this, we rectify our mistake of directly trying to optimize
the embeddings with jus loss and embedding lookup.
1. Load the embeddings taken out from gensim, [VOCAB_SIZE, EMBED_DIMS]
2. Normalize the embeddings
3. Get the wordIndexVocab and indexWordVocab
4. Build function to select triplet as batch function
5. Use one hot encoding function to encode in one-hot vectors and 
hstack them together
6. build triplet loss function
7. Build network
	-Make the 3 hstacked one hot vectors as placeholder
	-Make a dense layer with manually creation of Wx +b
	- IMP - initialize the weights with the embeddings from gensim
	(Here the idea is to train the embeddings as a weight matrix further)
	- Add the loss function and keep record
	- Final obtained weight matrix is the final embedding
	- Use eval to take it out, and also try to make  projector
'''
import tensorflow as tf
import os
import numpy as np
import math
import pickle
import random
import special_characters as preprocessor
from tensorflow.contrib.tensorboard.plugins import projector


EMBEDDINGS_SAVE_PATH = 'trained_word2vec_huge_vocab/dhcl_word2vec_model_300_embeddings.npy'
DICTI_DUMP_PATH = 'trained_word2vec_huge_vocab/dhcl_word2vec_model_300_vocabs_dicti_dump.dicti'

DATA_DICTI_PATH = 'demand_supply_resume_relavancy_data_project_skills.dicti'

# loading trained embedding from gensim to initialize this model
initialEmbeddings = np.load(EMBEDDINGS_SAVE_PATH)

#converting to tensor
initialEmbeddingTensor = tf.convert_to_tensor(initialEmbeddings, dtype=tf.float32)

#normalizing
norm = tf.sqrt(tf.reduce_sum(tf.square(initialEmbeddingTensor), 1, keep_dims=True))
normalized_embeddings = initialEmbeddingTensor / norm

# getting vocab and inverse-vocab
with open(DICTI_DUMP_PATH, 'rb') as f:
	wordIndexVocab, indexWordVocab = pickle.load(f)

# loading data for triplet
# we will load the data dicti containing demands as keys and supply skills
# in 3_4 category and 1_2 category 
with open(DATA_DICTI_PATH, 'rb') as f:
	dataDicti = pickle.load(f)

#hyper parameters for model

BATCH_SIZE = 1   #kept one as not sure about loss working
EMBED_DIMS = 300
EPOCHS = 1000  # as we are not going for batch 
SEED = 2
SAVE_DIR = os.path.join(os.getcwd(), 'saved_model_huge_vocab')
LOG_DIR = SAVE_DIR
EMBED_SAVE = os.path.join(SAVE_DIR, 'improved_embeddings_with_triplet_loss_huge_vocab.npy')
VOCAB_SIZE = len(wordIndexVocab)

random.seed(SEED)
def skills_to_number(skills, vocab):
	try:
		converted = [vocab[skill] for skill in skills]
	except KeyError:
		converted = [0, 0, 0] # here for this loss will become zero and no train
		print('not in vocab')
	return np.array(converted)

def one_hot_encoder(labelsList, class_number = VOCAB_SIZE):
	# this fn will return one-hot vectors for all values present
	# in labelsList horizontally stacked
	n = len(labelsList)
	out = np.zeros((n, class_number))
	out[range(n), labelsList] = 1
	return out

def get_triplet_batch(dataDicti, batchSize):

	# we will generate only one triplet at a time, not sure how
	# the triplet loss works in tensorflow
	# randomly selecting a demand

	#only where the supply 3_4 and supply 1_2 not empty
	validDemandsList = [key for key in dataDicti if len(dataDicti[key][2]) >= 2 and dataDicti[key][3] != [] and len(dataDicti[key][3]) != len(dataDicti[key][4])]

	# remove those keys where the 1_2 skills are same as intersection skills
	
	randomDemand = random.choice(validDemandsList)

	# from list of 3_4 skills for this demand, we will pick one skill
	# wanting to make it as anchor
	# we use replace because of new trained_word2vec

	anchorsList = []
	for a in dataDicti[randomDemand][2]:

		string = a.strip()
		string = preprocessor.replace_plus(string)
		string = preprocessor.replace_plus1(string)
		string = preprocessor.replace_plus2(string)
		string = preprocessor.replace_hash(string)
		string = preprocessor.replace_hash1(string)
		#string = preprocessor.replace_dot_net(string)
		string = string.replace('.net', ' dot net')
		#string = string.replace('.', ' ')
		string = string.replace(" ", "_")

		anchorsList.append(string)

	anchor = random.choice(anchorsList)
	# from list of 1_2 skills for this demand, we will pick one skill
	# wanting to make it as negative if it is not in intersection
	negativesList = []
	for a in dataDicti[randomDemand][3]:
		if a not in dataDicti[randomDemand][2]:
			string = a.strip()
			string = preprocessor.replace_plus(string)
			string = preprocessor.replace_plus1(string)
			string = preprocessor.replace_plus2(string)
			string = preprocessor.replace_hash(string)
			string = preprocessor.replace_hash1(string)
			#string = preprocessor.replace_dot_net(string)
			string = string.replace('.net', ' dot net')
			#string = string.replace('.', ' ')
			string = string.replace(" ", "_")

			negativesList.append(string)

	negative = random.choice(negativesList)

	#negative = random.choice([s.strip().replace(" ", "_") for s in dataDicti[randomDemand][3] if s not in dataDicti[randomDemand][2]])	
	# from list of 3_4 skills for this demand, we will pick one skill other
	# than anchor
	# wanting to make it as positive
	positivesList = []
	for a in dataDicti[randomDemand][2]:
		if a != anchor:
			string = a.strip()
			string = preprocessor.replace_plus(string)
			string = preprocessor.replace_plus1(string)
			string = preprocessor.replace_plus2(string)
			string = preprocessor.replace_hash(string)
			string = preprocessor.replace_hash1(string)
			#string = preprocessor.replace_dot_net(string)
			string = string.replace('.net', ' dot net')
			#string = string.replace('.', ' ')
			string = string.replace(" ", "_")

			positivesList.append(string)

	positive = random.choice(positivesList)

	#positive = random.choice([s.strip().replace(" ", "_") for s in dataDicti[randomDemand][2] if s != anchor])

	labels = [anchor, positive, negative]

	labels = skills_to_number(labels, wordIndexVocab)
	labels = one_hot_encoder(labels)

	return labels

def triplet_loss_custom(encodings, margin = 0.1):

	# after multiplying [3xV] input with [VxN] weight
	# we will get [3xN] encodings

	anchor = encodings[0]
	positive = encodings[1]
	negative = encodings[2]

	# taken from facenet gitub

	# Step 1: Compute the (embedding) distance between the anchor and the positive, 
	#you will need to sum over axis=-1
	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
	# Step 2: Compute the (encoding) distance between the anchor and the negative,
	# you will need to sum over axis=-1
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
	# Step 3: subtract the two previous distances and add margin.
	basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
	# Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
	
	triplet_loss = tf.reduce_sum(tf.maximum(basic_loss, tf.constant(0.0)))

	return triplet_loss

# building network to train the embeddings
with tf.name_scope('placeholder') as scope:
	#inp one hot vectors of anchor, positive, negative hstacked
	labelsPlaceholder = tf.placeholder(shape=[3, VOCAB_SIZE], dtype=tf.float32, name='labels')

with tf.name_scope('network') as scope:

	weights = tf.get_variable(shape=[VOCAB_SIZE, EMBED_DIMS], name='embed_weights')
	# just wanna update weights, not biases
	# this weights will get updated which will be initialized with 
	# gensim normalized embeddings and give out embedding as
	# [3xVOCAB_SIZE].[VOCA_SIZE X EMBED_DIMS] = [3XEMBED_DIMS]<-(encoding)
	# due to one-hot only the vector which is required will be calculated
	#rest will be zero, so it's corresponding embed will b fetched
	encodings = tf.matmul(labelsPlaceholder, weights)

with tf.name_scope('loss') as scope:

	loss = triplet_loss_custom(encodings)

	# add this loss to summary
	tf.summary.scalar('loss', loss)

with tf.name_scope('optimizer') as scope:

	optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	# this optimizer will update the weights

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

	writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
	init.run()

	# we have to initialize the embeddings from the embedding vectors
	# saved from gensim
	# running to assign intialize once in beggining
	sess.run(weights.assign(normalized_embeddings))

	for i in range(EPOCHS):

		batchX = get_triplet_batch(dataDicti, BATCH_SIZE)
		feedDicti = {labelsPlaceholder : batchX}

		_, l = sess.run([optimizer, loss], feed_dict = feedDicti)

		if i % 200 == 0:
			saver.save(sess, os.path.join(SAVE_DIR, 'triplet_loss_tf_model.ckpt'))
		print('Iter: '+str(i)+' triplet_Loss: '+"{:.6f}".format(l))

	finalEmbeddings = weights.eval()
	# saving this final embedding
	np.save(EMBED_SAVE, finalEmbeddings)
	print('final embeddings saved at: ', EMBED_SAVE)
	# Write corresponding labels for the embeddings.
	with open(LOG_DIR + '/metadata.tsv', 'w') as f:
		for i in range(VOCAB_SIZE):
				f.write(indexWordVocab[i].replace("\n", "") + '\n')
		# this file will be used by projector to take out labels,
		# '\n' replace is done because if the word is having \n, adding extra 
		#'\n' will create extra blank lines which will be misinterpreted

	# Create a configuration for visualizing embeddings with the labels in TensorBoard.
	config = projector.ProjectorConfig()
	embedding_conf = config.embeddings.add()
	embedding_conf.tensor_name = weights.name
	embedding_conf.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
	projector.visualize_embeddings(writer, config)
