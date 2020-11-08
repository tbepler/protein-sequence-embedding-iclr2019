from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .comparison import L1, pad_gap_scores


class SCOPCM(nn.Module):
    def __init__(self, embedding, similarity_kwargs={},
                 cmap_kwargs={}):
        super(SCOPCM, self).__init__()

        self.embedding = embedding
        embed_dim = embedding.nout

        self.scop_predict = OrdinalRegression(5, **similarity_kwargs)
        self.cmap_predict = ConvContactMap(embed_dim, **cmap_kwargs)

    def clip(self):
        self.scop_predict.clip()
        self.cmap_predict.clip()

    def forward(self, x):
        return self.embedding(x)

    def score(self, z_x, z_y):
        return self.scop_predict(z_x, z_y)

    def predict(self, z):
        return self.cmap_predict(z)


class ConvContactMap(nn.Module):
    def __init__(self, embed_dim, hidden_dim=50, width=7, act=nn.ReLU()):
        super(ConvContactMap, self).__init__()
        self.hidden = nn.Conv2d(2*embed_dim, hidden_dim, 1)
        self.act = act
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width//2)
        self.clip()

    def clip(self):
        # force the conv layer to be transpose invariant
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5*(w + w.transpose(2,3))

    def forward(self, z):
        return self.predict(z)

    def predict(self, z):
        # z is (b,L,d)
        z = z.transpose(1, 2) # (b,d,L)
        z_dif = torch.abs(z.unsqueeze(2) - z.unsqueeze(3))
        z_mul = z.unsqueeze(2)*z.unsqueeze(3)
        z = torch.cat([z_dif, z_mul], 1)
        # (b,2d,L,L)
        h = self.act(self.hidden(z))
        logits = self.conv(h).squeeze(1)
        return logits
    

class OrdinalRegression(nn.Module):
    def __init__(self, n_classes, compare=L1()
                , align_method='ssa', beta_init=10
                , allow_insertions=False, gap_init=-10
                ):
        super(OrdinalRegression, self).__init__()
        
        self.n_out = n_classes

        self.compare = compare
        self.align_method = align_method
        self.allow_insertions = allow_insertions
        self.gap = nn.Parameter(torch.FloatTensor([gap_init]))

        self.theta = nn.Parameter(torch.ones(1,n_classes-1))
        self.beta = nn.Parameter(torch.zeros(n_classes-1)+beta_init)
        self.clip()

    def clip(self):
        # clip the weights of ordinal regression to be non-negative
        self.theta.data.clamp_(min=0)

    def forward(self, z_x, z_y):

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




