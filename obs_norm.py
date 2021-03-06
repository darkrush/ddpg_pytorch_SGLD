import torch
from utils import *


class Run_Normalizer():
    def __init__(self, shape, if_cuda):
        self.n = torch.zeros(shape,dtype = torch.float)
        self.mean = torch.zeros(shape,dtype = torch.float)
        self.mean_diff = torch.zeros(shape,dtype = torch.float)
        self.var = torch.zeros(shape,dtype = torch.float)
        self.if_cuda = if_cuda
        if if_cuda:
            self.cuda()

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std
        
        
    def cuda(self):
        self.n=self.n.cuda()
        self.mean=self.mean.cuda()
        self.mean_diff=self.mean_diff.cuda()
        self.var=self.var.cuda()