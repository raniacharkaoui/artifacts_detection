import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import pandas as pd
import umap.umap_ as umap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde

path = sys.argv[1] 


name = path.split('/')[-2][:23]
model = path.split('/')[-3]


columns = ['tile', 'wsi'] + list(range(512)) 
latent = pd.read_csv(path, names=columns, index_col='tile')
latent.drop(columns=['wsi'], inplace = True)
print("Read latent space")

reducer = umap.UMAP()
embedding = reducer.fit_transform(latent)
print("UMAP is done")
print(f"The umap space embedding is of shape {embedding.shape}.")

"""
plt.figure(figsize =(18.5, 10.5)) 
plt.scatter(
    embedding[:, 0],
    embedding[:, 1], s=5, c='plum')
plt.gca().set_aspect('equal', 'datalim')
#plt.title('UMAP space of several images sampled', fontsize=24)
plt.savefig(f'{name}_umap_space.png',transparent=True)

print("UMAP space image is done")

#sample = latent.sample(n=55000)
#paths = list(sample.index)
paths = list(latent.index)

def getImage(path):
    return OffsetImage(plt.imread(path), zoom = 0.1)

x = embedding[:, 0]
y = embedding[:, 1]
fig, ax = plt.subplots()
fig.set_size_inches(17.5, 13.5)
ax.scatter(x, y) 
#fig = plt.figure(figsize =(7, 6)) 

for x0, y0, path in zip(x, y,paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)
plt.savefig(f'{name}_umap_space_with_tiles.png',transparent=True)
print("UMAP with tiles image is done")  
"""
# Calculate the point density
xy = np.vstack([embedding[:, 0],embedding[:, 1]])
z = gaussian_kde(xy)(xy) #density 

print("KDE is over")  
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
sc = ax.scatter(embedding[:, 0],embedding[:, 1],c=z, s=20)
plt.colorbar(sc)
plt.gca().set_aspect('equal', 'datalim')
#plt.title(f'{name} Kernel Density Plot', fontsize=24)
plt.savefig(f'{name}_{model}_density_plot.png',transparent=True)
plt.show()

#tile X Y density
density_df = pd.DataFrame()
density_df['tile'] = pd.Series(latent.index)

density_df.index = density_df['tile']
density_df.drop(columns=['tile'], inplace = True)

density_df['X'] = embedding[:, 0]
density_df['Y'] = embedding[:, 1]
density_df['density'] = z


low = density_df.loc[density_df['density'] <= 0.010]
low = low.sort_values(by='density')
print(f"The lowest density are of shape {low.shape}")

density_df.to_csv(f"/mnt/iribhm/people/rcharkao/data/density_{name}_{model}.csv")

def show_img(files, batch_number=0, save=False, fig_name="ranked_tiles.png"):
    plt.figure(figsize= (10,10))
    
    i=0
    for loc in range(25):
        plt.subplot(5,5,i+1)
        sample = plt.imread(files[loc + batch_number*25])
        plt.title(f'Density:{float(low.iloc[loc + batch_number*25, 2]):.2f}')
        plt.axis("off")
        plt.imshow(sample)
        i+=1
        
    if save:
        plt.savefig(fig_name,transparent=True)
        

print("Generating images with lowest density tiles")
files = list(low.index)
show_img(files, batch_number=0, save=True, fig_name= f"{name}_{model}_kde_0.png")
show_img(files, batch_number=1, save=True, fig_name= f"{name}_{model}_kde_1.png")
show_img(files, batch_number=2, save=True, fig_name= f"{name}_{model}_kde_2.png")
show_img(files, batch_number=3, save=True, fig_name= f"{name}_{model}_kde_3.png")
print("OVER")