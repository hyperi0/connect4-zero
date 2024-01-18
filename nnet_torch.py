import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
    
class PolicyNet(nn.Module):
    def __init__(self, rows=6, columns=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc = nn.Linear(32 * (columns - 4) * (rows - 4), 64)
        self.pi_fc = nn.Linear(64, columns)
        self.pi_sm = nn.Softmax(dim=1)
        self.v = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        pi = self.pi_sm(self.pi_fc(x))
        v = self.v(x)
        return pi, v
    
class ExperienceDataset(Dataset):
    def __init__(self, examples, device='cpu'):
        s, pi, v = zip(*examples)
        self.s = torch.tensor(s, dtype=torch.float).unsqueeze(1).to(device) # single channel input for convnet
        self.pi = torch.tensor(pi, dtype=torch.float).to(device)
        self.v = torch.tensor(v, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, i):
        return {'s': self.s[i], 'pi': self.pi[i], 'v': self.v[i]}

class Policy():
    def __init__(self, device='cpu'):
        self.device = device
        self.nnet = PolicyNet().to(device)
        
    def train(self, examples, batch_size=64, epochs=5, lr=1e-2):
        train_dataset = ExperienceDataset(examples, device=self.device)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=lr)
        loss_fn_pi = torch.nn.CrossEntropyLoss()
        loss_fn_v = torch.nn.MSELoss()
        for i in range(epochs):
            for batch in train_dataloader:
                s, pi, v = batch['s'], batch['pi'], batch['v']
                pi_pred, v_pred = self.nnet(s)
                loss = loss_fn_pi(pi_pred, pi) + loss_fn_v(v_pred.squeeze(), v)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, s):
        input = torch.tensor(s, dtype=torch.float).reshape((1,1,6,7)).to(self.device)
        pi, v = self.nnet(input)
        return pi[0].tolist(), v[0].item()