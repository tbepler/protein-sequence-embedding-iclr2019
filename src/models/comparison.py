from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1(nn.Module):
    def forward(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1)-y), -1)


class L2(nn.Module):
    def forward(self, x, y):
        return -torch.sum((x.unsqueeze(1)-y)**2, -1)


class DotProduct(nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y.t())


def pad_gap_scores(s, gap):
    col = gap.expand(s.size(0), 1)
    s = torch.cat([s, col], 1)
    row = gap.expand(1, s.size(1))
    s = torch.cat([s, row], 0)
    return s


class OrdinalRegression(nn.Module):
    def __init__(self, embedding, n_classes, compare=L1()
                , align_method='ssa', beta_init=10
                , allow_insertions=False, gap_init=-10
                ):
        super(OrdinalRegression, self).__init__()
        
        self.embedding = embedding
        self.n_out = n_classes

        self.compare = compare
        self.align_method = align_method
        self.allow_insertions = allow_insertions
        self.gap = nn.Parameter(torch.FloatTensor([gap_init]))

        self.theta = nn.Parameter(torch.ones(1,n_classes-1))
        self.beta = nn.Parameter(torch.zeros(n_classes-1)+beta_init)
        self.clip()

    def forward(self, x):
        return self.embedding(x)

    def clip(self):
        # clip the weights of ordinal regression to be non-negative
        self.theta.data.clamp_(min=0)

    def score(self, z_x, z_y):

        if self.align_method == 'ssa':
            s = self.compare(z_x, z_y)
            if self.allow_insertions:
                s = pad_gap_scores(s, self.gap)

            a = F.softmax(s, 1)
            b = F.softmax(s, 0)

            if self.allow_insertions:
                index = s.size(0)-1
                index = s.data.new(1).long().fill_(index)
                a = a.index_fill(0, index, 0)

                index = s.size(1)-1
                index = s.data.new(1).long().fill_(index)
                b = b.index_fill(1, index, 0)

            a = a + b - a*b
            c = torch.sum(a*s)/torch.sum(a)

        elif self.align_method == 'ua':
            s = self.compare(z_x, z_y)
            c = torch.mean(s)

        elif self.align_method == 'me':
            z_x = z_x.mean(0)
            z_y = z_y.mean(0)
            c = self.compare(z_x.unsqueeze(0), z_y.unsqueeze(0)).squeeze(0)

        else:
            raise Exception('Unknown alignment method: ' + self.align_method)

        logits = c*self.theta + self.beta
        return logits.view(-1)



