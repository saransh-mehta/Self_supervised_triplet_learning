import gensim
from gensim.models import Word2Vec
fname = "dhcl_word2vec_model_300"
model = Word2Vec.load(fname)
while True:
	try:
		print(model.wv.most_similar(positive=[raw_input('enter word : ')], topn=20))
	except Exception as e:
		print(e)
