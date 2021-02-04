#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
from tqdm import tqdm
import copy
import scipy.misc

from vae import VAE
from image_dataset import ImageDataset
from matplotlib import pyplot as plt


# In[2]:


########## Arguments ##########
dim = 64
latent_size = 512
channels = 3

batch_size = 128
num_epoch = 50
lr = 0.0001
lambda_kl = 0.01

best_model_path = "./vae_lambda001.pth"
val_img_path = "./images/val_img001.png"
#############################


# In[3]:


dataset = ImageDataset("train")
data_loader = DataLoader(dataset,  batch_size=batch_size, shuffle=True)

val_dataset = ImageDataset("val")
val_data_loader = DataLoader(
    val_dataset,  batch_size=batch_size, shuffle=False)


# In[4]:


def cal_loss(recon_x, x, mu, logvar):
    mse_loss = torch.mean((recon_x - x)**2)
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return mse_loss, kld_loss


# In[5]:


model = VAE(d=dim, zsize=latent_size, channels=channels)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# In[6]:


optimizer = optim.Adam(model.parameters(), lr=lr,
                       betas=(0.5, 0.999), weight_decay=1e-5)
best_loss = 99
best_model = None


# In[7]:


mse_losses = []
kld_losses = []

train_mse_losses = []
train_kld_losses = []

val_mse_losses = []
val_kld_losses = []

for epoch in range(num_epoch):
    print("{}/{}".format(epoch + 1, num_epoch))
    epoch_mse_loss = 0.0
    epoch_kld_loss = 0.0
    epoch_val_mse_loss = 0.0
    epoch_val_kld_loss = 0.0

    model.train()
    data_bar = tqdm(data_loader)
    for img in data_bar:
        optimizer.zero_grad()
        img = img.to(device)

        recon_img, mu, logvar = model(img)

        mse_loss, kld_loss = cal_loss(recon_img, img, mu, logvar)
#         loss = mse_loss
        loss = mse_loss + (kld_loss * lambda_kl)
        loss.backward()
        optimizer.step()

        epoch_mse_loss += mse_loss.item()
        epoch_kld_loss += kld_loss.item()

        mse_losses.append(mse_loss.item())
        kld_losses.append(kld_loss.item())
        data_bar.set_postfix(
            {'mse_loss': mse_loss.item(), 'kld_loss': kld_loss.item()})

    training_mse_loss = epoch_mse_loss / (len(dataset) / batch_size)
    training_kld_loss = epoch_kld_loss / (len(dataset) / batch_size)
#     training_mse_loss = (epoch_mse_loss / (3 * 64 * 64)) / (len(dataset))
#     training_kld_loss = (epoch_kld_loss / (3 * 64 * 64)) / (len(dataset))

    print("training_mse_loss: {}".format(training_mse_loss))
    print("training_kld_loss: {}".format(training_kld_loss))
    train_mse_losses.append(training_mse_loss)
    train_kld_losses.append(training_kld_loss)

    # validation
    model.eval()
    for img in tqdm(val_data_loader):
        optimizer.zero_grad()
        img = img.to(device)

        with torch.no_grad():
            recon_img, mu, logvar = model(img)
            mse_loss, kld_loss = cal_loss(recon_img, img, mu, logvar)
            loss = mse_loss + kld_loss * lambda_kl

        epoch_val_mse_loss += mse_loss.item()
        epoch_val_kld_loss += kld_loss.item()

    val_mse_loss = epoch_val_mse_loss / (len(val_dataset) / batch_size)
    val_kld_loss = epoch_val_kld_loss / (len(val_dataset) / batch_size)
#     val_mse_loss = (epoch_val_mse_loss / (3 * 64 * 64)) / (len(val_dataset))
#     val_kld_loss = (epoch_val_kld_loss / (3 * 64 * 64)) / (len(val_dataset))

    print("val_mse_loss: {}".format(val_mse_loss))
    print("val_kld_loss: {}".format(val_kld_loss))

    val_mse_losses.append(val_mse_loss)
    val_kld_losses.append(val_kld_loss)

    if (val_mse_loss + lambda_kl * val_kld_loss) < best_loss:
        best_model = copy.deepcopy(model.state_dict())
        # save img for reference
        recon_img = recon_img[-1].cpu().detach().numpy()
        recon_img = np.transpose(recon_img, (1, 2, 0))
        scipy.misc.imsave(val_img_path, recon_img)

    np.save('mse_loss_001.npy', np.array(mse_losses))
    np.save('kld_loss_001.npy', np.array(kld_losses))

    np.save('train_mse_losses_001.npy', np.array(train_mse_losses))
    np.save('train_kld_losses_001.npy', np.array(train_kld_losses))

    np.save('val_mse_loss_001.npy',  np.array(val_mse_losses))
    np.save('val_kld_loss_001.npy', np.array(val_kld_losses))


# In[ ]:


if best_model != None:
    torch.save(best_model, best_model_path)
    print("Model saved")
