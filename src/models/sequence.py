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


    def transform(self, z, last_only=False):
        # sequences are flanked by the start/stop token as:
        # [stop, x, stop]

        idx = [i for i in range(z.size(1)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(z.device)
        z_rvs = z.index_select(1, idx)

        z = z[:,:-1]
        z_rvs = z_rvs[:,:-1]
        idx = [i for i in range(z_rvs.size(1)-1, -1, -1)]
        idx = torch.LongTensor(idx).to(z_rvs.device)
    
        if last_only:
            if self.tied:
                h_fwd = z
                h_rvs = z_rvs
                for rnn in self.rnn:
                    h_fwd,_  = rnn(h_fwd)
                    h_fwd = self.dropout(h_fwd)
                    h_rvs,_ = rnn(h_rvs)
                    h_rvs = self.dropout(h_rvs)
            else:
                h_fwd = z
                h_rvs = z_rvs
                for lrnn,rrnn in zip(self.lrnn, self.rrnn): 
                    h_fwd,_  = lrnn(h_fwd)
                    h_fwd = self.dropout(h_fwd)
                    h_rvs,_ = rrnn(h_rvs)
                    h_rvs = self.dropout(h_rvs)
            hidden = (h_fwd, h_rvs.index_select(1, idx))
        else:
            hidden = []
            if self.tied:
                h_fwd = z
                h_rvs = z_rvs
                for rnn in self.rnn:
                    h_fwd,_  = rnn(h_fwd)
                    h_fwd = self.dropout(h_fwd)
                    h_rvs,_ = rnn(h_rvs)
                    h_rvs = self.dropout(h_rvs)
                    hidden.append((h_fwd, h_rvs.index_select(1, idx)))
            else:
                h_fwd = z
                h_rvs = z_rvs
                for lrnn,rrnn in zip(self.lrnn, self.rrnn): 
                    h_fwd,_  = lrnn(h_fwd)
                    h_fwd = self.dropout(h_fwd)
                    h_rvs,_ = rrnn(h_rvs)
                    h_rvs = self.dropout(h_rvs)
                    hidden.append((h_fwd, h_rvs.index_select(1, dx)))

        return hidden


    def encode(self, x):
        packed = type(x) is PackedSequence
        if packed:
            # pad with the start/stop token
            x,batch_sizes = pad_packed_sequence(x, batch_first=True, padding_value=self.mask_idx-1)
        x = x + 1
        ## append start/stop tokens to x
        x_ = x.data.new(x.size(0), x.size(1)+2).zero_()
        x_[:,1:-1] = x
        x = x_


        # sequences x are flanked by the start/stop token as:
        # [stop, x, stop]

        z = self.embed(x)
        hidden = self.transform(z)

        concat = []
        for h_fwd,h_rvs in hidden:
            h_fwd = h_fwd[:,:-1]
            h_rvs = h_rvs[:,1:]
            concat.append(h_fwd)
            concat.append(h_rvs)
        
        h = torch.cat(concat, 2)
        if packed:
            h = pack_padded_sequence(h, batch_sizes, batch_first=True)

        return h 


    def forward(self, x):
        packed = type(x) is PackedSequence
        if packed:
            # pad with the start/stop token
            x,batch_sizes = pad_packed_sequence(x, batch_first=True, padding_value=self.mask_idx)

        # sequences x are flanked by the start/stop token as:
        # [stop, x, stop]

        z = self.embed(x)

        h_fwd,h_rvs = self.transform(z, last_only=True)

        b = z.size(0)
        n = z.size(1) - 1

        h_fwd = h_fwd.contiguous()
        h_flat = h_fwd.view(b*n, h_fwd.size(2))
        logp_fwd = self.linear(h_flat)
        logp_fwd = logp_fwd.view(b, n, -1)

        zero = h_fwd.data.new(b,1,logp_fwd.size(2)).zero_()
        logp_fwd = torch.cat([zero, logp_fwd], 1)

        h_rvs = h_rvs.contiguous()
        logp_rvs = self.linear(h_rvs.view(-1, h_rvs.size(2))).view(b, n, -1)
        logp_rvs = torch.cat([logp_rvs, zero], 1)

        logp = F.log_softmax(logp_fwd + logp_rvs, dim=2)
        if packed:
            logp = pack_padded_sequence(logp, batch_sizes, batch_first=True)

        return logp 



