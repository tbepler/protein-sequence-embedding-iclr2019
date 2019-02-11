from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class LMEmbed(nn.Module):
    def __init__(self, nin, nout, lm, padding_idx=-1, transform=nn.ReLU()
                , sparse=False):
        super(LMEmbed, self).__init__()

        if padding_idx == -1:
            padding_idx = nin-1

        self.lm = lm
        self.embed = nn.Embedding(nin, nout, padding_idx=padding_idx, sparse=sparse)
        self.proj = nn.Linear(lm.hidden_size(), nout)
        self.transform = transform
        self.nout = nout

    def forward(self, x):
        packed = type(x) is PackedSequence
        h_lm = self.lm.encode(x)

        # embed and unpack if packed
        if packed:
            h = self.embed(x.data)
            h_lm = h_lm.data
        else:
            h = self.embed(x)

        # project
        h_lm = self.proj(h_lm)
        h = self.transform(h + h_lm)

        # repack if needed
        if packed:
            h = PackedSequence(h, x.batch_sizes)

        return h


class Linear(nn.Module):
    def __init__(self, nin, nhidden, nout, padding_idx=-1,
                 sparse=False, lm=None):
        super(Linear, self).__init__()
        
        if padding_idx == -1:
            padding_idx = nin-1
        
        if lm is not None:
            self.embed = LMEmbed(nin, nhidden, lm, padding_idx=padding_idx, sparse=sparse)
            self.proj = nn.Linear(self.embed.nout, nout)
            self.lm = True
        else:
            self.proj = nn.Embedding(nin, nout, padding_idx=padding_idx, sparse=sparse)
            self.lm = False

        self.nout = nout
        
        
    def forward(self, x):
        
        if self.lm:
            h = self.embed(x)
            if type(h) is PackedSequence:
                h = h.data
                z = self.proj(h)
                z = PackedSequence(z, x.batch_sizes)
            else:
                h = h.view(-1, h.size(2))
                z = self.proj(h)
                z = z.view(x.size(0), x.size(1), -1)
        else:
            if type(x) is PackedSequence:
                z = self.embed(x.data)
                z = PackedSequence(z, x.batch_sizes)
            else:
                z = self.embed(x)
    
        return z


class StackedRNN(nn.Module):
    def __init__(self, nin, nembed, nunits, nout, nlayers=2, padding_idx=-1, dropout=0,
                 rnn_type='lstm', sparse=False, lm=None):
        super(StackedRNN, self).__init__()
        
        if padding_idx == -1:
            padding_idx = nin-1
        
        if lm is not None:
            self.embed = LMEmbed(nin, nembed, lm, padding_idx=padding_idx, sparse=sparse)
            nembed = self.embed.nout
            self.lm = True
        else:
            self.embed = nn.Embedding(nin, nembed, padding_idx=padding_idx, sparse=sparse)
            self.lm = False

        if rnn_type == 'lstm':
            RNN = nn.LSTM
        elif rnn_type == 'gru':
            RNN = nn.GRU

        self.dropout = nn.Dropout(p=dropout)
        if nlayers == 1:
            dropout = 0

        self.rnn = RNN(nembed, nunits, nlayers, batch_first=True
                      , bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(2*nunits, nout)
        self.nout = nout

        
        
    def forward(self, x):
        
        if self.lm:
            h = self.embed(x)
        else:
            if type(x) is PackedSequence:
                h = self.embed(x.data)
                h = PackedSequence(h, x.batch_sizes)
            else:
                h = self.embed(x)
            
        h,_ = self.rnn(h)
        
        if type(h) is PackedSequence:
            h = h.data
            h = self.dropout(h)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = h.view(-1, h.size(2))
            h = self.dropout(h)
            z = self.proj(h)
            z = z.view(x.size(0), x.size(1), -1)
    
        return z


