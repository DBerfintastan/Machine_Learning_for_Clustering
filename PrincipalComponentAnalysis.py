# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:47:59 2022

@author: dberf
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

data=pd.read_csv("Hitters.csv")
data.dropna(inplace=True)
data=data._get_numeric_data()

data=StandardScaler().fit_transform(data)
pca=PCA(n_components=2)
pca_fit=pca.fit_transform(data)
pca_data=pd.DataFrame(data=pca_fit, columns=["birinci_bilesen","ikinci_bilesen"])

print("Aciklanan varyans:" ,pca.explained_variance_ratio_)

pca=PCA().fit(data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bilesen Sayisi")
plt.ylabel("Kumulatif Varyans Orani")