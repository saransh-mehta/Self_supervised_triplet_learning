'''
This is the first file in creating a precise word-embedding containing the skills 
which will be used for predicting inferred skills from given primary key skills

The embeddings will be precise in a way that:
1.  We are precisely picking up only skills to add in vocab. No other thing
2. After training word2vec on skills, we strive to improve the embeddings
through Triplet loss function. As we have recruiter marked supply resume which 
were not relevant to their corresponding demand, we use this as negative samples 
and the ones which are marked relevant to demand as positive sample.
Hence, decreasing distance between relevant, and increasing distance between
irrelevant skills

In this file, we are taking out all skills together from out restructured json 
to build vocab
'''
import numpy as np
import json
import pickle

JSON_PATH = '../demand_supply_restructured_extracted_keyskills.json'
SAVE_PATH = 'triplet_loss_embedding_skills.npy'
SAVE_PATH_2 = 'triplet_loss_embedding_skills.pickle'
SEED = 2

# reading json
def read_json(jsonPath):
	with open(JSON_PATH, 'r') as f:
	    dictiList = json.loads(f.read())
	print('json read')
	return dictiList

def list_shuffle(skillList, seed):
	# this function will shuffle the skills in the list so
	# that we can generate good n-gram pairs
	np.random.seed(seed)
	np.random.shuffle(skillList)
	# numpy random shuffle shuffles in places, returns None
	# hence no need to return
	print('shuffled')

def get_skills_from_dicti(dicti):

	# this function extracts primary skill (demand) from one json dicti
	# and put them together

	demandSkills = [s['skill'].lower() for s in dicti['primary_skills']]
	print('demand skill extracted')
	# it will always happen that project 'resume_rec_project_keyskills' field
	# will be there, cause it data is not available, we have placed [] for it
	projectSkills = [ p[0].lower() for p in dicti['resume_rec_project_keyskills']]
	print('project skill extracted')
	combinedSkillList = demandSkills + projectSkills
	#combinedSkillList = np.array(combinedSkillList)
	print('combined')
	return combinedSkillList

def save_np_array(arr, path):
	np.save(path, arr)
	print('array saved at: ', path)

def pickle_dump(path, data):
	with open(path, 'wb') as f:
		pickle.dump(data, f)
	print('pickle saved')

# doing it for all
dictiList = read_json(JSON_PATH)
allSkillsList = []
for i, dicti in enumerate(dictiList):
	print(i)
	#print(dicti, 'bla')
	sList = get_skills_from_dicti(dicti)
	#print(sList)
	#list_shuffle(sList, SEED)
	#print(sList)
	allSkillsList.append(sList)

allSkillsList = np.array(allSkillsList)
#save_np_array(allSkillsList, SAVE_PATH)
pickle_dump(SAVE_PATH_2, allSkillsList)