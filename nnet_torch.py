import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
    
class ResidualBlock(nn.Module):
    def __init__(self, n_channels=32, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x += residual
        x = F.relu(x)
        return x
    
class PolicyNet(nn.Module):
    def __init__(self, rows=6, columns=7, n_channels=32, n_res_blocks=3):
        super().__init__()
        # convolutional block
        self.conv = nn.Conv2d(1, n_channels, 3, padding='same')
        self.conv_bn = nn.BatchNorm2d(n_channels)
        # residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(n_res_blocks):
            self.res_blocks.append(ResidualBlock(n_channels=n_channels))
        # policy head
        self.policy_conv = nn.Conv2d(n_channels, 2, 1, padding='same')
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(n_channels * columns * rows / 4, columns)
        # value head
        self.value_conv = nn.Conv2d(n_channels, 1, padding='same')
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc = nn.Linear(n_channels * columns * rows, 256)
        self.value_out = nn.Linear(256, 1)

    def forward(self, x):
        # convolutional block
        x = self.conv(x)
        x = self.conv_bn(x)
        x = F.relu(x)
        # residual blocks
        x = self.res_blocks(x)
        # policy head
        pi = self.policy_conv(x)
        pi = self.policy_bn(pi)
        pi = F.relu(pi)
        pi = self.policy_fc(pi)
        pi = F.log_softmax(pi, dim=1)
        # value head
        v = self.value_conv(x)
        v = self.value_bn(x)
        v = F.relu(v)
        v = self.value_fc(v)
        v = F.relu(v)
        v = self.value_out(v)
        v = torch.tanh(v)
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
    
    def loss_fn_pi(self, pi_target, pi_pred):
        return -torch.sum(pi_target * pi_pred) / pi_target.size()[0]
    
    def train(self, examples, batch_size=64, epochs=5, lr=1e-2):
        train_dataset = ExperienceDataset(examples, device=self.device)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=lr)
        loss_fn_v = torch.nn.MSELoss()
        for i in range(epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                s, pi, v = batch['s'], batch['pi'], batch['v']
                pi_pred, v_pred = self.nnet(s)
                loss = self.loss_fn_pi(pi, pi_pred) + loss_fn_v(v_pred.squeeze(), v)
                loss.backward()
                optimizer.step()

    def predict(self, s):
        input = torch.tensor(s, dtype=torch.float).reshape((1,1,6,7)).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(input)
            return torch.exp(pi)[0].tolist(), v[0].item()