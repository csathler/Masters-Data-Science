import sys
import numpy as np
import pandas as pd
import matplotlib as mlp
if sys.platform == 'darwin':
     # in mac os only
     mlp.use('TkAgg')
import matplotlib.pyplot as plt
df = pd.read_csv("plotdata.data",index_col='algo')
df2 = df.copy()
df2.drop(['F1','ROC'],axis=1, inplace=True)
df2[df2.dataset=='TRAIN'].plot(kind='bar',ylim=(0.4,1),rot=22.5,fontsize=8,
                               grid=True,title="Best Performances in Training Dataset")
plt.legend(loc=2)
plt.savefig('train.png')
df2[df2.dataset=='TEST'].plot(kind='bar',ylim=(0.4,1),rot=22.5,fontsize=8,
                              grid=True,title="Best Performances in Test Dataset")
plt.legend(loc=2)
plt.savefig('test.png')
