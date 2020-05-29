import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class black1(nn.Module):

    def __init__(self, k=5, classes=10):
        super(black1, self).__init__()
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

        #get complement element
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0).cuda()
        Yg_ = torch.masked_select(yHat, y_zerohot.bool()).reshape(-1,self.classes -1)

        #generate random index
        ind = torch.randperm(self.classes -1)                             #sample w/o replacement
        # ind = torch.randint(0, self.classes -1, (self.classes -1,))     #sample with replacement
        complement = Yg_[:,ind[0:self.k]]

        #compute weighted softmax
        complement = self.k * torch.exp(complement)
        Yg = self.k* torch.exp(Yg)
        out = torch.cat((Yg, complement), 1)
        out = out/(out.sum(dim=1).unsqueeze(-1))

        #calculate blackout loss
        out_c = 1 - out
        mask = torch.zeros_like(yHat)[:,:self.k+1]
        mask[:,0] = 1
        mask_c = 1 - mask

        loss = -1. * (torch.log(out + 1e-10)*mask + 0.1*torch.log(out_c + 1e-10 )*mask_c).sum(-1).mean()  #
        return loss


class black2(nn.Module):

    def __init__(self, k=5, classes=10):
        super(black2, self).__init__()
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

        # #get complement element
        # y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
        #     1, y.view(self.batch_size, 1).data.cpu(), 0).cuda()
        # Yg_ = torch.masked_select(yHat, y_zerohot.bool()).reshape(-1,self.classes -1)

        #generate random index
        # ind = torch.randperm(self.classes)[:self.k].cuda()                          #sample w/o replacement
        ind = torch.randint(0, self.classes, (self.k,))     #sample with replacement
        complement = yHat[:,ind]

        m = torch.ones_like(complement)
        for i in range(m.size(0)):
            j = (y[i] == ind)
            m[i,j] = 0
        #compute weighted softmax
        complement = self.k * torch.exp(complement)
        complement *= m
        Yg = self.k* torch.exp(Yg)
        out = torch.cat((Yg, complement), 1)
        out = out/(out.sum(dim=1).unsqueeze(-1))

        #calculate blackout loss
        out_c = 1 - out
        mask = torch.zeros_like(yHat)[:,:self.k+1]
        mask[:,0] = 1
        mask_c = 1 - mask

        loss = -1. * (torch.log(out + 1e-10)*mask + torch.log(out_c + 1e-10 )*mask_c).sum(-1).mean()  #
        return loss