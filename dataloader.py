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

from network import Net

def cal_acc(y_true,y_pred):
    predict=y_pred.ge(.5).view(-1)
    return (y_true==predict).sum().float()/len(y_true)

def round_tensor(t,decimal_places=3):
    return round(t.item(),decimal_places)

randseed=42
model_path='model.pth'

data = pd.read_excel('./business_circle.xls')
data = data.dropna(how='any')
data.insert(5,'result',value=0,allow_duplicates=False)
data.loc[data[data.日均人流量 > 1500].index.tolist(),'result']=1
print(data.head(10))
X = data[['工作日上班时间人均停留时间','凌晨人均停留时间','周末人均停留时间','日均人流量']]
Y = data['result']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=randseed)

x_train = torch.from_numpy(x_train.to_numpy()).float()
y_train = torch.from_numpy(y_train.to_numpy()).float()
x_test  = torch.from_numpy(x_test.to_numpy()).float()
y_test  = torch.from_numpy(y_test.to_numpy()).float()

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

net = Net(x_train.shape[1])
criterion=torch.nn.BCELoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_train=x_train.to(device)
y_train=y_train.to(device)
x_test=x_test.to(device)
y_test=y_test.to(device)

net=net.to(device)
criterion=criterion.to(device)

for epoch in range(1000):
    y_pred=net(x_train)
    y_pred=torch.squeeze(y_pred)
    train_loss=criterion(y_pred,y_train)
    if epoch%100==0:
        train_acc= cal_acc(y_train,y_pred)
        y_test_pred=net(x_test)
        y_test_pred=torch.squeeze(y_test_pred)
        test_loss=criterion(y_test_pred,y_test)
        test_acc=cal_acc(y_test_pred,y_test)
        print(f'''epoch {epoch}
                train set-loss:{round_tensor(train_loss)},acc:{round_tensor(train_acc)}
                test set-loss:{round_tensor(test_loss)},acc:{round_tensor(test_acc)}''')
        torch.save(net,model_path)
        
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

