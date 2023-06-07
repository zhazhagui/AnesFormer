import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler
import copy





class Solver:
    def __init__(self, args):
        self.args = args

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.sampler = sampler.AdversarySampler(self.args.budget)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                # for img, label, _ in dataloader:
                #     yield img, label
                for batch in dataloader:
                    img, label, subject_index = batch['data'], batch['label'], batch['subject_index']
                    yield img, label, subject_index

        else:
            while True:
                # for img, _, _ in dataloader:
                #     yield img
                for batch in dataloader:
                    img, label, subject_index = batch['data'], batch['label'], batch['subject_index']
                    yield img, subject_index




    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             self.args.cuda)

        return querry_indices
                


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
    
    def aae_loss(self, x, recon):
        MSE = self.mse_loss(recon, x)
        return MSE
