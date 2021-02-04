#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


# In[2]:


class VAE(nn.Module):
    def __init__(self, d, zsize, channels=3):
        super(VAE, self).__init__()
        
        self.d = d
        self.zsize = zsize
        self.channels = channels
        
        # encoder
        self.e1 = nn.Conv2d(channels, d, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(d)
        
        self.e2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(d * 2)
        
        self.e3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(d * 4)
        
        self.e4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(d * 8)        

        self.fc1 = nn.Linear(d * 8 * 4 * 4, zsize)
        self.fc2 = nn.Linear(d * 8 * 4 * 4, zsize)
        
        # decoder
        self.d1 = nn.Linear(zsize, d * 8 * 4 * 4)

        self.d2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.dbn2 = nn.BatchNorm2d(d * 4)

        self.d3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.dbn3 = nn.BatchNorm2d(d * 2)

        self.d4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.dbn4 = nn.BatchNorm2d(d)

        self.d5 = nn.ConvTranspose2d(d, channels, 4, 2, 1)

    def encode(self, x):
        h1 = F.relu(self.bn1(self.e1(x)))
        h2 = F.relu(self.bn2(self.e2(h1)))
        h3 = F.relu(self.bn3(self.e3(h2)))
        h4 = F.relu(self.bn4(self.e4(h3)))
        h4 = h4.view(h4.shape[0], self.d * 8 * 4 * 4)

        return self.fc1(h4), self.fc2(h4)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = z.view(z.shape[0], self.zsize)
        h1 = self.d1(z)
        h1 = h1.view(h1.shape[0], self.d * 8, 4, 4)
        #x = self.deconv1_bn(x)
        h1 = F.leaky_relu(h1, 0.2)
        
        h2 = F.leaky_relu(self.dbn2(self.d2(h1)), 0.2)
        h3 = F.leaky_relu(self.dbn3(self.d3(h2)), 0.2)
        h4 = F.leaky_relu(self.dbn4(self.d4(h3)), 0.2)
        h5 = torch.tanh(self.d5(h4))

        return h5

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar



# In[ ]:




