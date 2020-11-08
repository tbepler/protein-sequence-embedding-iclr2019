from __future__ import print_function, division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_labels(s):
    y = np.zeros(len(s), dtype=int)
    for i in range(len(s)):
        if s[i:i+1] == b'I':
            y[i] = 0
        elif s[i:i+1] == b'O':
            y[i] = 1
        elif s[i:i+1] == b'M':
            y[i] = 2
        elif s[i:i+1] == b'S':
            y[i] = 3
        else:
            raise Exception('Unrecognized annotation: ' + s[i:i+1].decode('utf-8'))
    return y


def transmembrane_regions(y):
    regions = []
    start = -1
    for i in range(len(y)):
        if y[i] == 2 and start < 0:
            start = i
        elif y[i] != 2 and start > 0:
            regions.append((start,i))
            start = -1
    if start > 0:
        regions.append((start, len(y)))
    return regions


def is_prediction_correct(y_hat, y):
    ## prediction is correct if it has the same number of transmembrane regions
    ## and those overlap real transmembrane regions by at least 5 bases
    ## and it starts with a signaling peptide when y does
    pred_regions = transmembrane_regions(y_hat)
    target_regions = transmembrane_regions(y)
    if len(pred_regions) != len(target_regions):
        return 0
    
    for p,t in zip(pred_regions, target_regions):
        if p[1] <= t[0]:
            return 0
        if t[1] <= p[0]:
            return 0
        s = max(p[0], t[0])
        e = min(p[1], t[1])
        overlap = e - s
        if overlap < 5:
            return 0
        
    # finally, check signal peptide
    if y[0] == 3 and y_hat[0] != 3:
        return 0
    
    return 1
    

## TOPCONS uses a very specific state architecture for HMM
## we can adopt this to describe the transmembrane grammar
class Grammar:
    def __init__(self, n_helix=21, signal_helix=True):
        ## describe the transmembrane states
        n_states = 3 + 2*n_helix
        
        start = np.zeros(n_states)
        start[0] = 1.0 # inner
        start[1] = 1.0 # outer
        start[2] = 1.0 # signal peptide
        
        end = np.zeros(n_states)
        end[0] = 1.0 # from inner
        end[1] = 1.0 # from outer
        
        trans = np.zeros((n_states, n_states))
        trans[0,0] = 1.0 # inner -> inner
        trans[0,3] = 1.0 # inner -> helix (i->o)
        trans[1,1] = 1.0 # outer -> outer
        trans[1,3+n_helix] = 1.0 # outer -> helix (o->i)
        
        trans[2,0] = 1.0 # signal -> inner
        trans[2,1] = 1.0 # signal -> outer
        
        for i in range(3,2+n_helix): # i->o helices
            trans[i,i+1] = 1.0
        trans[2+n_helix,1] = 1.0 # helix (i->o) -> outer
        
        for i in range(3+n_helix,2+2*n_helix): # o->i helices
            trans[i,i+1] = 1.0
        trans[2+2*n_helix,0] = 1.0 # helix (o->i) -> inner
        
        emit = np.zeros((n_states, 4))
        emit[0,0] = 1.0 # inner
        emit[0,1] = 1.0
        emit[1,0] = 1.0 # outer
        emit[1,1] = 1.0
        emit[2,3] = 1.0 # signal peptide
        #if signal_helix:
        #    emit[2,2] = 1.0
        for i in range(3,3+2*n_helix): # helices
            emit[i,2] = 1.0
        
        mapping = np.zeros(n_states, dtype=int)
        mapping[0] = 0
        mapping[1] = 1
        mapping[2] = 3
        mapping[3:3+2*n_helix] = 2
        
        self.start = np.log(start) - np.log(start.sum())
        self.end = np.log(end) - np.log(end.sum())
        self.trans = np.log(trans) - np.log(trans.sum(1, keepdims=True))
        self.emit = emit
        self.mapping = mapping
        
    def decode(self, logp):
        p = np.exp(logp)
        z = np.log(np.dot(p, self.emit.T))
        
        tb = np.zeros(z.shape, dtype=np.int8) - 1
        p0 = z[0] + self.start
        for i in range(z.shape[0] - 1):
            trans = p0[:,np.newaxis] + self.trans + z[i+1] #
            tb[i+1] = np.argmax(trans, 0)
            p0 = np.max(trans, 0)
        # transition to end
        p0 = p0 + self.end
        state = np.argmax(p0)
        score = np.max(p0)
        # traceback most likely sequence of states
        y = np.zeros(z.shape[0], dtype=int)
        j = state
        y[-1] = j
        for i in range(z.shape[0]-1, 0, -1):
            j = tb[i,j]
            y[i-1] = j
            
        # map the states
        y = self.mapping[y]
            
        return y, score
        
    def predict_viterbi(self, xs, model, use_cuda=False):
        y_hats = []
        with torch.no_grad():
            for x in xs:
                if use_cuda:
                    x = x.cuda()
                log_p_hat = F.log_softmax(model(x), 1).cpu().numpy()
                y_hat,_ = self.decode(log_p_hat)
                y_hats.append(y_hat)
        return y_hats
        
        
        
        

