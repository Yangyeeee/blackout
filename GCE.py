import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
# For MNIST dataset we difine classes to 10
classes = 10


class GuidedComplementEntropy(nn.Module):

    def __init__(self, alpha, classes=10):
        super(GuidedComplementEntropy, self).__init__()
        self.alpha = alpha
        self.classes = classes
    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)

        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        # avoiding numerical issues (second)
        guided_factor = (Yg + 1e-7) ** self.alpha
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (third)  torch.log(guided_factor.squeeze())
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()


        guided_output = guided_factor.squeeze()* torch.sum(output, dim=1)
        loss = torch.sum(guided_output)
        loss /= float(self.batch_size)
        loss /= math.log(float(self.classes))

        # guided_output = guided_factor.squeeze() * torch.sum(output, dim=1)
        # loss = torch.sum(guided_output)
        # loss /= math.log(float(self.classes))
        # loss = torch.log(-1.0 * loss)
        # loss /= float(self.batch_size)
        # loss = -1.0 * loss

        # guided_output = guided_factor.squeeze() * torch.sum(output, dim=1)
        # guided_output /= float(self.batch_size)
        # loss = torch.sum(guided_output)
        # loss /= math.log(float(self.classes))
        # loss = torch.log(-1.0 * loss)
        # loss = -1.0 * loss
        return  loss


class black(nn.Module):

    def __init__(self, k=5, classes=10):
        super(black, self).__init__()
        self.k = k
        self.classes = classes
    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        # yHat = torch.exp(yHat)
        maxxx = torch.max(yHat, dim=-1)[0].unsqueeze(-1)
        yHat = yHat - maxxx
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))


        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0).cuda()
        Yg_ = torch.masked_select(yHat, y_zerohot.bool()).reshape(-1,9)
        ind = torch.randint(0, 9, (9,)) #torch.randperm(9)
        Yg_ = Yg_[:,ind]
        # Yg_ = Yg * y_zerohot.cuda()
        # sorted, _ = torch.sort(Yg_,dim=-1)
        m = torch.tensor([1 for i in range(self.k)] + [0 for j in range(self.k,9)]).unsqueeze(0).cuda()
        y_zerohot1 = torch.ones(self.batch_size, self.classes-1).cuda()*m
        complement = torch.masked_select(Yg_, y_zerohot1.bool()).reshape(-1,self.k)

        complement = 1./self.k * torch.exp(complement)
        Yg = 0.8 * torch.exp(Yg)
        out = torch.cat((Yg, complement), 1)

        out = out/(out.sum(dim=1).unsqueeze(-1) + 1e-7)

        out1 = 1 - out
        mask = torch.tensor([1]+[0*i for i in range(self.k)]).unsqueeze(0).cuda()
        mask1 = 1 - mask

        loss = -1. * (torch.log(out + 1e-7)*mask + torch.log(out1 + 1e-7)*mask1).sum(-1).mean()

        return loss

class blackout0(nn.Module):

    def __init__(self, k=5, classes=10, eps=1e-10, use_cuda=False):
        super(blackout0, self).__init__()
        self.k = k
        self.classes = classes
        self.eps = eps
        self.use_cuda = use_cuda

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        # yHat = torch.exp(yHat)
        maxxx = torch.max(yHat, dim=-1)[0].unsqueeze(-1).detach()
        yHat = yHat - maxxx
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))

        #get complement element
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(1, y.view(self.batch_size, 1).data.cpu(), 0)
        if self.use_cuda:
            y_zerohot = y_zerohot.cuda()
        Yg_ = torch.masked_select(yHat, y_zerohot.bool()).reshape(-1,self.classes -1)

        #generate random index
        #ind = torch.randperm(self.classes -1)                          #sample w/o replacement
        ind = torch.randint(0, self.classes -1, (self.batch_size, self.k))     #sample with replacement
        if self.use_cuda:
            ind = ind.cuda()
        complement = Yg_.gather(1, ind)

        #compute weighted softmax
        complement = self.k * torch.exp(complement)
        Yg = self.k * torch.exp(Yg)
        out = torch.cat((Yg, complement), 1)
        out = out / (out.sum(dim=1).unsqueeze(-1))

        #calculate blackout loss
        out_c = 1 - out
        mask = torch.zeros_like(yHat)[:,:self.k+1]
        mask[:,0] = 1
        mask_c = 1 - mask

        loss = -1. * (torch.log(out + self.eps) * mask + torch.log(out_c + self.eps) * mask_c).mean()  #
        return loss


class blackout1(nn.Module):

    def __init__(self, k=5, classes=10, eps=1e-10, use_cuda=False):
        super(blackout1, self).__init__()
        self.k = k
        self.classes = classes
        self.eps = eps
        self.use_cuda = use_cuda

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        # yHat = torch.exp(yHat)
        maxxx = torch.max(yHat, dim=-1)[0].unsqueeze(-1).detach()
        yHat = yHat - maxxx
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))

        #get complement element
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(1, y.view(self.batch_size, 1).data.cpu(), 0)
        if self.use_cuda:
            y_zerohot = y_zerohot.cuda()
        Yg_ = torch.masked_select(yHat, y_zerohot.bool()).reshape(-1,self.classes -1)

        #generate random index
        #ind = torch.randperm(self.classes -1)                          #sample w/o replacement
        ind = torch.randint(0, self.classes - 1, (self.k,))     #sample with replacement
        if self.use_cuda:
            ind = ind.cuda()
        complement = Yg_[:,ind]

        #compute weighted softmax
        complement = self.k * torch.exp(complement)
        Yg = self.k * torch.exp(Yg)
        out = torch.cat((Yg, complement), 1)
        out = out / (out.sum(dim=1).unsqueeze(-1))

        #calculate blackout loss
        out_c = 1 - out
        mask = torch.zeros_like(yHat)[:,:self.k+1]
        mask[:,0] = 1
        mask_c = 1 - mask

        loss = -1. * (torch.log(out + self.eps) * mask + torch.log(out_c + self.eps) * mask_c).mean()  #
        return loss


class blackout2(nn.Module):

    def __init__(self, k=5, classes=10, eps=1e-10, use_cuda=False):
        super(blackout2, self).__init__()
        self.k = k
        self.classes = classes
        self.eps = eps
        self.use_cuda = use_cuda

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        # yHat = torch.exp(yHat)
        maxxx = torch.max(yHat, dim=-1)[0].unsqueeze(-1).detach()
        yHat = yHat - maxxx
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))

        # #get complement element
        # y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
        #     1, y.view(self.batch_size, 1).data.cpu(), 0).cuda()
        # Yg_ = torch.masked_select(yHat, y_zerohot.bool()).reshape(-1,self.classes -1)
        #generate random index
        # ind = torch.randperm(self.classes)[:self.k]        #sample w/o replacement
        ind = torch.randint(0, self.classes, (self.k,))     #sample with replacement
        if self.use_cuda:
            ind = ind.cuda()
        complement = yHat[:,ind]

        m = torch.ones_like(complement)
        for i in range(m.size(0)):
            j = (y[i] == ind)
            m[i,j] = 0
        #compute weighted softmax
        complement = self.k * torch.exp(complement)
        complement *= m
        Yg = self.k * torch.exp(Yg)
        out = torch.cat((Yg, complement), 1)
        out = out / (out.sum(dim=1).unsqueeze(-1))

        #calculate blackout loss
        out_c = 1 - out
        mask = torch.zeros_like(yHat)[:,:self.k+1]
        mask[:,0] = 1
        mask_c = 1 - mask

        loss = -1. * (torch.log(out + self.eps) * mask + torch.log(out_c + self.eps) * mask_c).mean() #.sum(-1).mean()  #
        return loss


class blackout3(nn.Module):

    def __init__(self, k=5, classes=10, eps=1e-10, use_cuda=False, p=0.5):
        super(blackout3, self).__init__()
        self.k = k
        self.classes = classes
        self.eps = eps
        self.use_cuda = use_cuda
        prob = generate_p_cifar100(p)
        self.eval_prob = prob.copy()
        np.fill_diagonal(prob, 0)
        sampling_prob = prob / (prob.sum(-1).reshape(-1, 1))
        self.sampling_prob = sampling_prob

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        maxxx = torch.max(yHat, dim=-1)[0].unsqueeze(-1).detach()
        yHat = yHat - maxxx
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))

        #complement = torch.zeros(self.batch_size, self.k)
        p = np.zeros((self.batch_size, self.k), dtype=np.single)
        q = np.zeros((self.batch_size, 1), dtype=np.single)
        ind = np.zeros((self.batch_size, self.k), dtype=np.int64)

        #generate random index
        for i in range(self.batch_size):
            ind[i] = np.random.choice(a=self.classes, size=self.k, replace=True, p=self.sampling_prob[y[i]])
            p[i] = self.eval_prob[y[i], ind[i]]
            q[i] = self.eval_prob[y[i], y[i]]
        ind = torch.from_numpy(ind)
        p = torch.from_numpy(p)
        q = torch.from_numpy(q)
        if self.use_cuda:
            ind = ind.cuda()
            p = p.cuda()
            q = q.cuda()

        # compute weighted softmax
        complement = yHat.gather(1, ind)
        complement = (1/p) * torch.exp(complement)
        Yg = (1/q) * torch.exp(Yg)

        out = torch.cat((Yg, complement), 1)
        out = out/(out.sum(dim=1).unsqueeze(-1))

        #calculate blackout loss
        out_c = 1 - out
        mask = torch.zeros_like(yHat)[:,:self.k+1]
        mask[:,0] = 1
        mask_c = 1 - mask

        loss = -1. * (torch.log(out + self.eps) * mask + torch.log(out_c + self.eps) * mask_c).mean()
        return loss


class blackout4(nn.Module):

    def __init__(self, k=5, classes=10, eps=1e-10, use_cuda=False,prob=0):
        super(blackout4, self).__init__()
        self.k = k
        self.classes = classes
        self.eps = eps
        self.use_cuda = use_cuda
        self.prob = prob
        # self.prob_c = prob.copy()
        # np.fill_diagonal(self.prob_c, self.prob_c.diagonal() * 0)

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        maxxx = torch.max(yHat, dim=-1)[0].unsqueeze(-1).detach()
        yHat = yHat - maxxx
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))


        complement = torch.zeros_like(yHat)[:, :self.k]
        p = torch.ones_like(yHat)[:, :self.k]
        #generate random index
        for i in range(yHat.shape[0]):
            ind = np.random.choice(a=100, size=self.k, replace=True, p=self.prob[y[i]])
            complement[i] = yHat[i,ind]
            p[i] *= p.new_tensor(1/self.prob[y[i]][ind])


        #compute weighted softmax
        complement = p *torch.exp(complement) #
        q = torch.min(p,dim=-1,keepdim=True)[0]
        Yg = q*torch.exp(Yg)
        out = torch.cat((Yg, complement), 1)
        out = out/(out.sum(dim=1).unsqueeze(-1))

        #calculate blackout loss
        out_c = 1 - out
        mask = torch.zeros_like(yHat)[:,:self.k+1]
        mask[:,0] = 1
        mask_c = 1 - mask

        loss = -1. * (torch.log(out + self.eps) * mask + torch.log(out_c + self.eps) * mask_c).mean() #.sum(-1).mean()  #
        return loss