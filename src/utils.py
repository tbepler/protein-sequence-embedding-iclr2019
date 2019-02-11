from __future__ import print_function,division

import numpy as np

import torch
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

def pack_sequences(X, order=None):
    
    #X = [x.squeeze(0) for x in X]
    
    n = len(X)
    lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]
    m = max(len(x) for x in X)
    
    X_block = X[0].new(n,m).zero_()
    
    for i in range(n):
        j = order[i]
        x = X[j]
        X_block[i,:len(x)] = x
        
    #X_block = torch.from_numpy(X_block)
        
    lengths = lengths[order]
    X = pack_padded_sequence(X_block, lengths, batch_first=True)
    
    return X, order


def unpack_sequences(X, order):
    X,lengths = pad_packed_sequence(X, batch_first=True)
    X_block = [None]*len(order)
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i,:lengths[i]]
    return X_block


def collate_lists(args):
    x = [a[0] for a in args]
    y = [a[1] for a in args]
    return x, y


class ContactMapDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment=None, fragment=False, mi=64, ma=500):
        self.X = X
        self.Y = Y
        self.augment = augment
        self.fragment = fragment
        self.mi = mi
        self.ma = ma
        """
        if fragment: # multiply sequence occurence by expected number of fragments
            lengths = np.array([len(x) for x in X])
            mi = np.clip(lengths, None, mi)
            ma = np.clip(lengths, None, ma)
            weights = 2*lengths/(ma + mi)
            mul = np.ceil(weights).astype(int)
            X_ = []
            Y_ = []
            for i,n in enumerate(mul):
                X_ += [X[i]]*n
                Y_ += [Y[i]]*n
            self.X = X_
            self.Y = Y_
        """

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.Y[i]
        if self.fragment and len(x) > self.mi:
            mi = self.mi
            ma = min(self.ma, len(x))
            l = np.random.randint(mi, ma+1)
            i = np.random.randint(len(x)-l+1)
            xl = x[i:i+l]
            yl = y[i:i+l,i:i+l]
            # make sure there are unmasked observations
            while torch.sum(yl >= 0) == 0:
                l = np.random.randint(mi, ma+1)
                i = np.random.randint(len(x)-l+1)
                xl = x[i:i+l]
                yl = y[i:i+l,i:i+l]
            y = yl.contiguous()
            x = xl
        if self.augment is not None:
            x = self.augment(x)
        return x, y


class AllPairsDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment=None):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)**2

    def __getitem__(self, k):
        n = len(self.X)
        i = k//n
        j = k%n

        x0 = self.X[i]
        x1 = self.X[j]
        if self.augment is not None:
            x0 = self.augment(x0)
            x1 = self.augment(x1)

        y = self.Y[i,j]
        #y = torch.cumprod((self.Y[i] == self.Y[j]).long(), 0).sum()

        return x0, x1, y


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, X0, X1, Y):
        self.X0 = X0
        self.X1 = X1
        self.Y = Y

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, i):
        return self.X0[i], self.X1[i], self.Y[i]


def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)


class MultinomialResample:
    def __init__(self, trans, p):
        self.p = (1-p)*torch.eye(trans.size(0)).to(trans.device) + p*trans

    def __call__(self, x):
        #print(x.size(), x.dtype)
        p = self.p[x] # get distribution for each x
        return torch.multinomial(p, 1).view(-1) # sample from distribution




