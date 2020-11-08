from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class BiLM(nn.Module):
    def __init__(self, nin, nout, embedding_dim, hidden_dim, num_layers
                , tied=True, mask_idx=None, dropout=0):
        super(BiLM, self).__init__()
        
        if mask_idx is None:
            mask_idx = nin-1
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(nin, embedding_dim, padding_idx=mask_idx)
        self.dropout = nn.Dropout(p=dropout)

        self.tied = tied
        if tied:
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.rnn = nn.ModuleList(layers)
        else:
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.lrnn = nn.ModuleList(layers)

            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.rrnn = nn.ModuleList(layers)

        self.linear = nn.Linear(hidden_dim, nout)

    def hidden_size(self):
        h = 0
        if self.tied:
            for layer in self.rnn:
                h += 2*layer.hidden_size
        else:
            for layer in self.lrnn:
                h += layer.hidden_size
            for layer in self.rrnn:
                h += layer.hidden_size
        return h

    
    def reverse(self, h):
        packed = type(h) is PackedSequence
        if packed:
            h,batch_sizes = pad_packed_sequence(h, batch_first=True)
            h_rvs = h.clone().zero_()
            for i in range(h.size(0)):
                n = batch_sizes[i]
                idx = [j for j in range(n-1, -1, -1)]
                idx = torch.LongTensor(idx).to(h.device)
                h_rvs[i,:n] = h[i].index_select(0, idx)
            # repack h_rvs
            h_rvs = pack_padded_sequence(h_rvs, batch_sizes, batch_first=True)
        else:
            idx = [i for i in range(h.size(1)-1, -1, -1)]
            idx = torch.LongTensor(idx).to(h.device)
            h_rvs = h.index_select(1, idx)
        return h_rvs


    def transform(self, z_fwd, z_rvs, last_only=False):
        # sequences are flanked by the start/stop token as:
        # [stop, x, stop]

        # z_fwd should be [stop,x]
        # z_rvs should be [x,stop] reversed

        # first, do the forward direction
        if self.tied:
            layers = self.rnn
        else:
            layers = self.lrnn

        h_fwd = []
        h = z_fwd
        for rnn in layers:
            h,_ = rnn(h)
            if type(h) is PackedSequence:
                h = PackedSequence(self.dropout(h.data), h.batch_sizes)
            else:
                h = self.dropout(h)
            if not last_only:
                h_fwd.append(h)
        if last_only:
            h_fwd = h

        # now, do the reverse direction
        if self.tied:
            layers = self.rnn
        else:
            layers = self.rrnn

        # we'll need to reverse the direction of these
        # hidden states back to match forward direction

        h_rvs = []
        h = z_rvs
        for rnn in layers:
            h,_ = rnn(h)
            if type(h) is PackedSequence:
                h = PackedSequence(self.dropout(h.data), h.batch_sizes)
            else:
                h = self.dropout(h)
            if not last_only:
                h_rvs.append(self.reverse(h))
        if last_only:
            h_rvs = self.reverse(h)

        return h_fwd,h_rvs


    def embed_and_split(self, x, pad=False):

        packed = type(x) is PackedSequence
        if packed:
            x,batch_sizes = pad_packed_sequence(x, batch_first=True)

        if pad:
            # pad x with the start/stop token
            x = x + 1
            ## append start/stop tokens to x
            x_ = x.data.new(x.size(0), x.size(1)+2).zero_()
            if packed:
                for i in range(len(batch_sizes)):
                    n = batch_sizes[i]
                    x_[i,1:n+1] = x[i,:n]
                batch_sizes = [s+2 for s in batch_sizes]
            else:
                x_[:,1:-1] = x
            x = x_

        # sequences x are flanked by the start/stop token as:
        # [stop, x, stop]

        # now, encode x as distributed vectors
        z = self.embed(x)

        # to pass to transform, we discard the last element for z_fwd and the first element for z_rvs
        z_fwd = z[:,:-1]
        z_rvs = z[:,1:]
        if packed:
            lengths = [s-1 for s in batch_sizes]
            z_fwd = pack_padded_sequence(z_fwd, lengths, batch_first=True)
            z_rvs = pack_padded_sequence(z_rvs, lengths, batch_first=True)
        # reverse z_rvs
        z_rvs = self.reverse(z_rvs)

        return z_fwd, z_rvs


    def encode(self, x):
        z_fwd,z_rvs = self.embed_and_split(x, pad=True)
        h_fwd_layers,h_rvs_layers = self.transform(z_fwd, z_rvs)

        # concatenate hidden layers together
        packed = type(z_fwd) is PackedSequence
        concat = []
        for h_fwd,h_rvs in zip(h_fwd_layers,h_rvs_layers):
            if packed:
                h_fwd,batch_sizes = pad_packed_sequence(h_fwd, batch_first=True)
                h_rvs,batch_sizes = pad_packed_sequence(h_rvs, batch_first=True)
            # discard last element of h_fwd and first element of h_rvs
            h_fwd = h_fwd[:,:-1]
            h_rvs = h_rvs[:,1:]
            
            # accumulate for concatenation
            concat.append(h_fwd)
            concat.append(h_rvs)
        
        h = torch.cat(concat, 2)
        if packed:
            batch_sizes = [s-1 for s in batch_sizes]
            h = pack_padded_sequence(h, batch_sizes, batch_first=True)

        return h 


    def forward(self, x):
        # x's are already flanked by the star/stop token as:
        # [stop, x, stop]
        z_fwd,z_rvs = self.embed_and_split(x, pad=False)

        h_fwd,h_rvs = self.transform(z_fwd, z_rvs, last_only=True)

        packed = type(z_fwd) is PackedSequence
        if packed:
            h_flat = h_fwd.data
            logp_fwd = self.linear(h_flat)
            logp_fwd = PackedSequence(logp_fwd, h_fwd.batch_sizes)

            h_flat = h_rvs.data
            logp_rvs = self.linear(h_flat)
            logp_rvs = PackedSequence(logp_rvs, h_rvs.batch_sizes)

            logp_fwd,batch_sizes = pad_packed_sequence(logp_fwd, batch_first=True)
            logp_rvs,batch_sizes = pad_packed_sequence(logp_rvs, batch_first=True)

        else:
            b = h_fwd.size(0)
            n = h_fwd.size(1)
            h_flat = h_fwd.contiguous().view(-1, h_fwd.size(2))
            logp_fwd = self.linear(h_flat)
            logp_fwd = logp_fwd.view(b, n, -1)

            h_flat = h_rvs.contiguous().view(-1, h_rvs.size(2))
            logp_rvs = self.linear(h_flat)
            logp_rvs = logp_rvs.view(b, n, -1)

        # prepend forward logp with zero
        # postpend reverse logp with zero

        b = h_fwd.size(0)
        zero = h_fwd.data.new(b,1,logp_fwd.size(2)).zero_()
        logp_fwd = torch.cat([zero, logp_fwd], 1)
        logp_rvs = torch.cat([logp_rvs, zero], 1)

        logp = F.log_softmax(logp_fwd + logp_rvs, dim=2)

        if packed:
            batch_sizes = [s+1 for s in batch_sizes]
            logp = pack_padded_sequence(logp, batch_sizes, batch_first=True)

        return logp 



