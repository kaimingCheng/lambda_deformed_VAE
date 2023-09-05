import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn

from vae import loss_function, loss_function_alpha, iwae, iwae_np, vr_max, loss_function_alpha_np


def show_images(images):
    images = torch.reshape(
        images, [images.shape[0], -1]
    )  # images reshape to (batch_size, D)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return

def one_hot(labels, class_size):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    - class_size: Scalar representing of target classes our dataset
    Outputs:
    - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when
    the ground truth label for image i is j, and targets[i, :j] &
    targets[i, j + 1:] are equal to 0
    """
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def train_vae(epoch, model, train_loader, alpha=0.9, cond=False):
    """
    Train a VAE!

    Inputs:
    - epoch: Current epoch number
    - model: VAE model object
    - train_loader: PyTorch Dataloader object that contains our training data
    - cond: Boolean value representing whether we're training a VAE
    """
    model.train()
    train_loss = 0
    num_classes = 10
    loss = None
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device="cuda")
        if cond:
            one_hot_vec = one_hot(labels, num_classes).to(device="cuda")
            recon_batch, mu, logvar, z = model(data, one_hot_vec)
        else:
            recon_batch, mu, logvar, z = model(data)
        optimizer.zero_grad()
        loss = loss_function_alpha(recon_batch, data, mu, logvar, z, alpha=alpha)
        # print(loss)
        loss.backward()
        train_loss += loss.data
        # for name, param in model.named_parameters():
          # if torch.isnan(param.grad).any():
              # print(param.grad)
              
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-4)
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, loss.data))
