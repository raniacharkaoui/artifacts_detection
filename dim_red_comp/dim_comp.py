import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import umap.umap_ as umap
import pacmap
import sys 
import time 

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
#from sklearn.neighbors import NearestNeighbors


start1 = time.time()
path = sys.argv[1] 


#X = pd.read_csv(path, index_col ='tile')

#Ciga 512 
#Resmlp 384
#Visformer 768
#eca_nfnet_l0 2304

col = ['tile', 'svs'] + list(range(768)) 
X = pd.read_csv(path, names = col, index_col='tile')
X.drop(columns=['svs'], inplace = True)
print(X.head())

def getImage(path):
    return OffsetImage(plt.imread(path), zoom = 0.1)

def displayEmbeddingAnnotation(embedding, algo, model, name):
    x = embedding[:, 0]
    y = embedding[:, 1]
    fig, ax = plt.subplots()
    fig.set_size_inches(17.5, 13.5)
    ax.scatter(x, y) 
    #fig = plt.figure(figsize =(7, 6)) 

    for x0, y0, path in zip(x, y,paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.savefig(f"/mnt/iribhm/people/rcharkao/first_examples_thyroid/res/dim_red_{name}__{algo}_{model}.png", transparent=True)

def displayDensity(embedding, algo, model, name):     
    # Calculate the point density
    xy = np.vstack([embedding[:, 0],embedding[:, 1]])
    z = gaussian_kde(xy)(xy) #density 

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    sc = ax.scatter(embedding[:, 0],embedding[:, 1],c=z, s=15)
    plt.colorbar(sc)
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig(f"/mnt/iribhm/people/rcharkao/first_examples_thyroid/res/kde_vis_{name}__{algo}_{model}.png", transparent=True)


pmap = pacmap.PaCMAP(n_components=2, n_neighbors=40) 
umap = umap.UMAP(n_components=2, n_neighbors=40)
pca = PCA(n_components=2)

name = path.split('/')[-2][:23]
model = path.split('/')[-3]

algos = [pmap, umap, pca]
paths = list(X.index)
sample = X.sample(n=10000)

for algo in algos:
    start = time.time()
    
    embedding = algo.fit_transform(sample)
    
    end = time.time()
    #print(f'It took {end - start}s to run {str(algo)[:6]}')
    with open('/mnt/iribhm/people/rcharkao/first_examples_thyroid/res_dim_red.txt', 'w') as f:
    f.write(f'It took {end - start}s to run {str(algo)[:6]} on {name}')
    f.write('\n')
    
    displayEmbeddingAnnotation(embedding, str(algo)[:6], model, name)
    displayDensity(embedding, str(algo)[:6], model, name)


end1 = time.time()

#print(f'Total time: {end1 - start1}s for {name} in {model}')
with open('/mnt/iribhm/people/rcharkao/first_examples_thyroid/res_dim_red.txt', 'w') as f:
    f.write(f'Total time: {end1 - start1}s for {name} in {model}')
    f.write('\n')
    