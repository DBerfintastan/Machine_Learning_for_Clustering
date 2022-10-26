# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:44:02 2022

@author: dberf
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

data=pd.read_csv("USArrests.csv", index_col=0)

hc_complete=linkage(data,"complete")
hc_average=linkage(data, "average")


plt.figure(figsize=(15,10))
plt.title("Dendrogram")
plt.xlabel("Gozlem Birimleri")
plt.ylabel("Uzakliklar")
dendrogram(hc_complete,
           leaf_font_size=10,
           truncate_mode="lastp",
           p=4,
           show_contracted=True)

plt.figure(figsize=(15,10))
plt.title("Dendrogram")
plt.xlabel("Gozlem Birimleri")
plt.ylabel("Uzakliklar")
dendrogram(hc_average,
           leaf_font_size=10,
           show_contracted=True)