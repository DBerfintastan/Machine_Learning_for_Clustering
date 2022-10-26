# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 16:34:38 2022

@author: dberf
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

data=pd.read_csv("USArrests.csv", index_col=0)

data.isnull().sum() #bos veri var mi?
data.describe().T #verinin ozeti
#data.hist(figsize=(10,10)) #grafige dokme

kmeans=KMeans(n_clusters=4) #kume sayisi
k_fit=kmeans.fit(data) 

k_fit.n_clusters 
k_fit.cluster_centers_ #kume merkezlerindeki gozlem birimleri
k_fit.labels_ #gozlem birimleri hangi kumelerde?

#KUMELERIN GORSELLESTIRILMESI
k_means=KMeans(n_clusters=2).fit(data)
kumeler=k_means.labels_
#plt.scatter(data.iloc[:,0], data.iloc[:,1], c=kumeler, s=50,cmap="viridis")

merkezler=k_means.cluster_centers_
#plt.scatter(merkezler[:,0], merkezler[:,1], c="black", s=200,alpha=0.5)

ssd=[] #uzaklık karelerinin toplamı
K=range(1,30)

for k in K:
    kmeans=KMeans(n_clusters=k).fit(data)
    ssd.append(kmeans.inertia_)
    
plt.plot(K,ssd,"bx-")
plt.xlabel("Farkli K degerlerine karsilik uzaklik artik toplamlari")
plt.title("Optimum kume sayisi icin elbow yontemi")
    
k_means_fitted=KMeans(n_clusters=4).fit(data)
kumeler=k_means_fitted.labels_
pd.DataFrame({"Eyaletler":data.index, "Kumeler":kumeler})

data["Kume_No"]=kumeler
