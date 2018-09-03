from __future__ import absolute_import, print_function, division
# we did the above to handle compatibility issues in py 2 and 3
import os
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
# below we will googlez word2vec from gensim
import gensim.models.word2vec as word2vec
import seaborn as sb  #to plot nicely
#we will also remove stopwords using nltk and
#punctuations using string module
from sklearn.manifold import TSNE
import pandas as pd

SEED = 0
MODEL_NAME = 'model.w2v'
MODEL_PATH = os.path.join(os.getcwd(), 'trained_word2vec',MODEL_NAME)
print('loading the saved trained model')

# NOTE : the direct methods word2vec.vocab and syn0 and other methods have been moved to
# KeyedVectored class.
model = word2vec.Word2Vec.load(MODEL_PATH)

print('moving to compressing the {} dimensional vector to 2D for plot')

tsne = TSNE(n_components=2, random_state= SEED)
#here we have just made the model, not feed data
#tsne is also a trained model, which compresses the vectors n generate co ordinates for them in 2D

allVectorMatrix = model.wv.syn0
# this is bring all the vectors to a vector matrix which can be fed to tsne
# the direct method syn0 has been moved to KeyedIndex class, hence
#we have to use wv

print('Training Tsne for compressing the vectors.May take time...')

allVectorMatrix_2d = tsne.fit_transform(allVectorMatrix)

print('Successfully compressed !')

#now we will go for plotting,
print('visualizing in dataframe')

df = pd.DataFrame(

    [ (word, coords[0], coords[1]) for word, coords in [(word, allVectorMatrix_2d[model.wv.vocab[word].index])
            for word in model.wv.vocab ]
    ],
    columns=["word", "x", "y"]
)

print(df.head(10))

print('now going for plotting using seaborn')

sb.set_context('poster')

#this is different from plt.scatter.
#this function is for plotting dataframe. Look documentation
df.plot.scatter('x', 'y', s=10, figsize = (20, 12))

#

print('the overall plot must be ready..,Thanks for patience ! Close the plot for next ')
plt.savefig('demand_supply_keyskills_all.png')
'''''''''''
def zoomInTo(xStart, xEnd, yStart, yEnd):
    axes = plt.axes()
    axes.set_xlim(xStart, xEnd)
    axes.set_ylim(yStart, yEnd)
    plt.scatter(df['x'], df['y'], s=35)
    
    #, figsize=(20, 12)
    plt.show()
    return
zoomInTo(xStart=-50.0, xEnd=0, yStart= -50.0, yEnd= 0)
'''


def plot_region(xStart, xEnd, yStart, yEnd, savePath ):

    region  =df[ (df.x >= xStart) & (df.x <= xEnd) & (df.y >= yStart) & (df.y <= yEnd) ]

    design = region.plot.scatter("x", "y", s=35, figsize=(10, 8))

    # iterrows() is used to iterate over dataFrame rows, returns both index anf=d values
    for index, point in region.iterrows():

        design.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=4)
    plt.savefig(savePath, dpi=300)

#x = input("enter x co - ordinates in tuple format")
#y = input("enter y co ordinate in tuple format")
s1 = 'demand_supply_keyskills_zoom_1.png'
s2 = 'demand_supply_keyskills_zoom_2.png'
s3 = 'demand_supply_keyskills_zoom_3.png'
s4 = 'demand_supply_keyskills_zoom_4.png'
s5 = 'demand_supply_keyskills_zoom_5.png'
plot_region(xStart= -55.0, xEnd = -35.0,yStart= -30.0, yEnd = 5.0, savePath=s1)
plot_region(xStart= -50.0, xEnd = -30.0,yStart= 5.0, yEnd = 20.0, savePath=s2)
plot_region(xStart= -10.0, xEnd = 25.0,yStart= 10.0, yEnd = 35.0, savePath=s3)
plot_region(xStart= -55.0, xEnd = -35.0,yStart= 50.0, yEnd = 70.0, savePath=s4)