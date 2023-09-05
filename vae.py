from __future__ import print_function
from ast import NameConstant

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.distributions.normal as N


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = None  # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        self.hidden_dim = 300
        self.encoder = nn.Sequential(nn.Flatten(),
        nn.Linear(input_size, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU()
        )
        self.mu_layer = nn.Linear(self.hidden_dim, latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, latent_size)
        
        self.decoder = nn.Sequential(
        nn.Linear(latent_size, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, input_size),
        nn.Sigmoid(),
        nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x, k=50):

        N = x.shape[0]
        x_hat = None
        mu = None
        logvar = None
        
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        mu = mu.repeat(k, 1)
        logvar = logvar.repeat(k, 1)

        z_list = []
        x_hat = []

        x = x.repeat(k, 1, 1, 1)
        z = reparametrize(mu, logvar)
        
        x_hat = self.decoder(z)

        # for i in range(k):
        #   z = reparametrize(mu, logvar)
        #   z_list.append(z)
        #   x_hat.append(self.decoder(z))
        
        # output = self.decoder(z)
        # mu1 = self.mu_layer1(z1)
        # logvar1 = self.logvar_layer1(z1)
        # x_hat1 = reparametrize(mu1, logvar1)
        # s = nn.Sigmoid()
        # x_hat = s(x_hat1)
        # x_hat = x_hat.view(128, 1, 28, 28)
        
        return x_hat, mu, logvar, z


def reparametrize(mu, logvar):
    """
    
    """
    z = None

    std = torch.exp(0.5*logvar) # cuz, log(std^2) = 2 log(std)
    eps = torch.randn_like(std) # standard gaussian
    z = mu + std * eps
    # print(z2)

    return z


def loss_function(x_hat, x, mu, logvar):
    """
    
    """
    loss = None
    N, Z = mu.shape
    
    loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') - 0.5*torch.sum(logvar-mu**2-torch.exp(logvar))-0.5*Z*N
    loss /= N

    return loss


def likelihood(data, mean, std):
    N, Z = data.shape
    mean = mean.to('cuda')
    std = std.to('cuda')
    # out = -torch.log(std) - (data - mean)**2 /2*std**2
    
    # l = torch.sum(out)
    
    out = (1/(2*torch.pi)**0.5)*(1/std) * torch.exp(-((data - mean)**2) /(2*std**2))
    
    out = torch.sum(torch.log(out), 1)
    # l = torch.sum(out)
    out = out.view(N, 1)
    # print(out)
    return out
    


def sample(x_hat, mu, logvar, z, alpha=1):

    stdvar = torch.exp(logvar/2)
    # stdvar1 = torch.exp(logvar1/2)
    # q = torch.normal(mean=mu, std=stdvar)
    z = reparametrize(mu, logvar)
    ql = likelihood(z, mu, stdvar)
    # x1 = x_hat.view(128, 28*28)
    # pl = likelihood(x_hat, mu1, stdvar1)
    pz = likelihood(z, torch.zeros(z.shape), torch.ones(z.shape))

    return (pz-ql)

def bcelossalpha(x, y):
    s1 = x.shape[0]
    s2 = y.shape[0]

    x = x.view(s1, 28*28)
    y = y.view(s2, 28*28)
    
    k = (y*torch.log(x)) + ((1-y) * torch.log(1-x))
    
    k = torch.sum(k, dim=1)
    k = -k
    k = k.view(s1, 1)
    # k = (-k / (28*28))
    #
    return k


def loss_function_alpha(x_hat, x, mu, logvar, z, alpha=0.3, k=50, mode='batch'):
    mu = mu.to('cuda')
    logvar = logvar.to('cuda')
    stdvar = torch.exp(logvar/2)
    N, Z = mu.shape
    # w = torch.ones(N/k, k)
    x = x.repeat(k, 1, 1, 1)

    # Vanilla VAE
    if alpha == 1:
      loss = -nn.functional.binary_cross_entropy(x_hat[0], x, reduction='sum') + torch.sum(sample1(mu, logvar, z))
      loss = -loss
      # loss /= N

    # Deformed VAE
    else:
      
      loss = 0

      if mode == 'loop':
        for i in range(k):
          
          loss_p = (-bcelossalpha(x_hat[i], x))  # log(p(x|z))
          # p(x|z)^(1-alpha)
          # (p(z)/q(z|x))^(1-alpha)
          
          loss_p += sample1(mu, logvar, z[i])
          loss_p *= (1-alpha)
          w[:, i] = loss_p[:, 0]
      
      if mode == 'batch':
        loss_p = -nn.functional.binary_cross_entropy(x_hat, x, reduce=False)
        loss_p = torch.sum(loss_p, (1, 2, 3))
        loss_p += sample1(mu, logvar, z)
        loss_p = loss_p.view(k, int(N/k))
        loss_p *= (1-alpha)
        
      loss = torch.logsumexp(loss_p, 0) - torch.log(torch.tensor(k))
      loss = loss.to('cuda')
      loss /= (1-alpha)
      loss = - torch.sum(loss)
    loss /= int(N/k)
    # print(loss)

    return loss

def iwae(x_hat, x, mu, logvar, z, k=2):
    N, Z = mu.shape
    stdvar = torch.exp(logvar/2)
    
    loss = 0
    for i in range(k):
      
      log_weights = loss_function(x_hat[i], x, mu, logvar)
      weights = torch.exp(log_weights)
      loss += weights
    
    
    loss = torch.log(loss)
    
    loss = -torch.sum(loss)
    loss /= N

    # print(loss)
    return loss


def iwae_np(x_hat, x, mu, logvar, z=1, k=2):
    N, Z = mu.shape
    stdvar = torch.exp(logvar/2)
    
    loss = 0
    for i in range(k):
      
      log_weights = loss_function(x_hat[i], x, mu, logvar).cpu().detach().numpy()
      # print(np.exp(-float(log_weights)))
      weights = np.exp(-float(log_weights))
      # print(weights)
      # print(np.exp(-500))
      loss += weights
    
    loss /= k
    loss = -np.log(loss)
    loss = torch.tensor(loss)
    loss.requires_grad = True
    # print(loss)
    return loss


def loss_function_alpha_np(x_hat, x, mu, logvar, z, alpha=0.99, k=2):
    stdvar = torch.exp(logvar/2)
    N, Z = mu.shape

    # Vanilla VAE
    if alpha == 1:
      loss = -nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') + torch.sum(sample(x_hat, mu, logvar, alpha))
      loss = -loss
      loss /= N

    # Deformed VAE
    else:
      
      loss = 0
      for i in range(k):
        
        loss_p = (-bcelossalpha(x_hat[i], x)).cpu().detach().numpy()  # log(p(x|z))
        loss_post = np.exp(float((1-alpha)*loss_p))
        # loss_post = torch.exp((1-alpha)*loss_p)  # p(x|z)^(1-alpha)
        # print(loss_post)
        loss_i = loss_post * np.exp((1-alpha)*float(sample(x_hat, mu, logvar, z[i], alpha).cpu().detach().numpy()))  # (p(z)/q(z|x))^(1-alpha)
        # print(sample(x_hat, mu, logvar, alpha))
        loss += loss_i
        
      
      loss /= k
      loss = np.log(loss)
      loss = -(1/(1-alpha)) * loss
      
      # print(loss)
      # loss = -np.sum(loss)
      # loss /= N


    print(loss)

    return loss


def vr_max(x_hat, x, mu, logvar, z=1, k=2):
    N, Z = mu.shape
    stdvar = torch.exp(logvar/2)
    
    loss = 1000
    for i in range(k):
      
      cur_loss = loss_function(x_hat[i], x, mu, logvar)
      if cur_loss < loss:
        loss = cur_loss
    
    # loss /= N
    
    return loss

def iwae_np1(x, vae_model, k=50, mode='batch'):
    # N, Z = mu.shape
    # stdvar = torch.exp(logvar/2)
    # w = torch.ones(N, k)
    x = x.repeat(k, 1, 1, 1)
    # img_expand = torch.expand(x, (k, 1, 128, 128))
    loss = 0
    # if mode == 'loop':
    #   for i in range(k):
    #     z = reparametrize(mu, logvar)
    #     x_hat = vae_model.decoder(z)
    #     loss_p = -nn.functional.binary_cross_entropy(x_hat, x, reduce=False) # log(p(x|z))
        
    #     loss_p = torch.sum(loss_p, (2, 3))
    #     loss_p += sample1(mu, logvar, z)
        
    #     w[:, i] = loss_p[:, 0]
    
    if mode == 'batch':
      x_hat, mu, logvar, z = vae_model(x, k=1)
      
      loss_p = -nn.functional.binary_cross_entropy(x_hat, x, reduce=False)
      loss_p = torch.sum(loss_p, (1, 2, 3))
      
      loss_p += sample1(mu, logvar, z)
      
    loss = torch.logsumexp(loss_p, 0) - torch.log(torch.tensor(k))
    loss = - torch.sum(loss)
    # print(loss)
    
    return loss

# def sample1(mu, logvar, z):

#     stdvar = torch.exp(logvar/2)
    
#     ql = likelihood(z, mu, stdvar)
    
#     pz = likelihood(z, torch.zeros(z.shape), torch.ones(z.shape))
    

#     return (pz-ql)

def sample1(mu, logvar, z):
    stdvar = torch.exp(logvar/2)

    dist_q = N.Normal(mu, stdvar)
    dist_p = N.Normal(0, 1)

    log_pz = dist_p.log_prob(z)
    log_qz = dist_q.log_prob(z)
    
    return torch.sum((log_pz - log_qz), 1)
