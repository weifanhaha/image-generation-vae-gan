#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np
import scipy.misc


# In[2]:


from models import Generator, Discriminator, weights_init
from image_dataset import ImageDataset


# In[3]:


batch_size = 64
lr = 0.0002
beta1 = 0.5
num_epochs = 50

nz = 100
ngf = 64
ndf = 64

best_model_path = "./best_G.pth"
fake_img_path = "./fake_img.png"


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator(nz=nz, ngf=ngf, nc=3)
netG = netG.to(device)
netG.apply(weights_init)
# print(netG)

netD = Discriminator(ndf=ndf, nc=3)
netD = netD.to(device)
netD.apply(weights_init)
# print(netD)


# In[5]:


dataset = ImageDataset("train")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[6]:


criterion = nn.BCELoss()

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
REAL_LABEL = 1
FAKE_LABEL = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# In[8]:


D_losses = []
G_losses = []

best_G_loss = 999

for epoch in range(num_epochs):
    print("Epoch {} / {}".format(epoch, num_epochs))

    epoch_D_loss = 0.0
    epoch_G_loss = 0.0

    loader = tqdm(dataloader)

    for imgs in loader:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        imgs = imgs.to(device)
        curr_batch_size = imgs.size(0)
        label = torch.full((curr_batch_size,), REAL_LABEL,
                           dtype=imgs.dtype, device=device)

        output = netD(imgs)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(curr_batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(FAKE_LABEL)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        epoch_D_loss += errD.item()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(REAL_LABEL)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        epoch_G_loss += errG.item()

        loader.set_postfix(
            {
                "Loss_D": errD.item(),
                "Loss_G": errG.item(),
                "D(x)": "%4f" % D_x,
                "D(G(z))": "%.4f / %.4f" % (D_G_z1, D_G_z2)
            }
        )

    training_D_loss = epoch_D_loss / (len(dataset) / batch_size)
    training_G_loss = epoch_G_loss / (len(dataset) / batch_size)

    print("training_D_loss: {}".format(training_D_loss))
    print("training_G_loss: {}".format(training_G_loss))

    D_losses.append(training_D_loss)
    G_losses.append(training_G_loss)

    if training_G_loss < best_G_loss:
        # save model
        best_model = copy.deepcopy(netG.state_dict())

        # save fig for reference
        fake_img = fake[-1].cpu().detach().numpy()
        fake_img = np.transpose(fake_img, (1, 2, 0))
        scipy.misc.imsave(fake_img_path, fake_img)

    np.save('training_G_losses.npy', np.array(G_losses))
    np.save('training_D_losses.npy', np.array(D_losses))


# In[ ]:


# if best_model != None:
#     torch.save(best_model, best_model_path)
#     print("Model saved")


# In[ ]:


# training_G_losses = np.load('training_G_losses.npy')
# training_D_losses = np.load('training_D_losses.npy')

# from matplotlib import pyplot as plt

# plt.plot(training_G_losses)
# plt.plot(training_D_losses)


# In[ ]:
