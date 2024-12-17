# import trimap
import umap

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd

class umapMethod:
    def __init__(self, xx,yy,n_neighbors=15,min_dist=0.1,n_components=2,metric='euclidean',n_jobs=1):
        self.xx = xx
        self.yy = yy
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.n_jobs = n_jobs
    def applyumap(self,catNames,axiss,titless):    
        scaled_xx = StandardScaler().fit_transform(self.xx)
        reducer = umap.UMAP(n_neighbors=self.n_neighbors,
                            min_dist=self.min_dist,
                            n_components=self.n_components,
                            metric=self.metric,
                            n_jobs=self.n_jobs)
        mapper = reducer.fit(scaled_xx)
        df = pd.DataFrame(np.concatenate((mapper.embedding_[:, 0].reshape(-1,1),mapper.embedding_[:, 1].reshape(-1,1),self.yy),axis=1),\
            columns=['x1','x2','labels'])
        sns.scatterplot(x='x1',y='x2',data=df,hue='labels',alpha=0.2,ax=axiss,legend='auto').set(title=titless)


     