import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,n_feature) -> None:
        super().__init__()
        self.fc1=torch.nn.Linear(n_feature,5)
        self.fc2=torch.nn.Linear(5,3)
        self.fc3=torch.nn.Linear(3,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))