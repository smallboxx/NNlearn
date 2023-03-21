import torch
import os
import numbers as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from torch import nn,optim
import torch.nn.functional as F

data = pd.read_excel('./business_circle.xls')
data = data.dropna(how='any')
data.insert(5,'result',value=0,allow_duplicates=False)
data.loc[data[data.]]
print(data.bhead(5))