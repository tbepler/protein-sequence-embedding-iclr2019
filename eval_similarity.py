from __future__ import print_function, division

import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from scipy.stats import pearsonr,spearmanr

from bepler.utils import pack_sequences, unpack_sequences
from bepler.alphabets import Uniprot21
from bepler.alignment import nw_score
from bepler.metrics import average_precision


def encode_sequence(x, alphabet):
    # convert to bytes and uppercase
    x = x.encode('utf-8').upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    return x


def load_pairs(path, alphabet):
    table = pd.read_csv(path, sep='\t')

    x0 = [encode_sequence(x, alphabet) for x in table['sequence_A']]
    x1 = [encode_sequence(x, alphabet) for x in table['sequence_B']]
    y = table['similarity'].values

    return x0, x1, y


class NWAlign:
    def __init__(self, alphabet):
        from Bio.SubsMat import MatrixInfo as matlist
        L = len(alphabet)
        subst = np.zeros((L,L), dtype=np.int32)
        for i in range(L):
            for j in range(i,L):
                a = alphabet[i]
                b = alphabet[j]
                subst[i,j] = subst[j,i] = matlist.blosum62[(b,a)]
        self.subst = subst
        self.gap = -11
        self.extend = -1

    def __call__(self, x, y):
        b = len(x)
        scores = np.zeros(b)
        for i in range(b):
            scores[i] = nw_score(x[i], y[i], self.subst, self.gap, self.extend)
        return scores


class TorchModel:
    def __init__(self, model, use_cuda, mode='ssa'):
        self.model = model
        self.use_cuda = use_cuda
        self.mode = mode

    def __call__(self, x, y):
        n = len(x)
        c = [torch.from_numpy(x_).long() for x_ in x] + [torch.from_numpy(y_).long() for y_ in y]

        c,order = pack_sequences(c)
        if self.use_cuda:
            c = c.cuda()

        with torch.no_grad():
            z = self.model(c) # embed the sequences
            z = unpack_sequences(z, order)

            scores = np.zeros(n)
            if self.mode == 'align':
                for i in range(n):
                    z_x = z[i]
                    z_y = z[i+n]

                    logits = self.model.score(z_x, z_y)
                    p = F.sigmoid(logits).cpu()
                    p_ge = torch.ones(p.size(0)+1)
                    p_ge[1:] = p
                    p_lt = torch.ones(p.size(0)+1)
                    p_lt[:-1] = 1 - p
                    p = p_ge*p_lt
                    p = p/p.sum() # make sure p is normalized
                    levels = torch.arange(5).float()
                    scores[i] = torch.sum(p*levels).item()

            elif self.mode == 'coarse':
                z_x = z[:n]
                z_y = z[n:]
                z_x = torch.stack([z.mean(0) for z in z_x], 0)
                z_y = torch.stack([z.mean(0) for z in z_y], 0)
                scores[:] = -torch.sum(torch.abs(z_x - z_y), 1).cpu().numpy()

        return scores

def find_best_threshold(x, y, tr0=-np.inf):
    order = np.argsort(x)
    
    tp = np.zeros(len(x)+1)
    tp[0] = y.sum()
    tn = np.zeros(len(x)+1)
    tn[0] = 0
    
    
    for i in range(len(x)):
        j = order[i]
        tp[i+1] = tp[i] - y[j] 
        tn[i+1] = tn[i] + 1 - y[j]
    
    acc = (tp + tn)/len(y)
    i = np.argmax(acc) - 1
    
    tr = x[order[i]]
    if i < 0:
        tr = tr0
    
    return tr


def find_best_thresholds(x, y):
    thresholds = np.zeros(5)
    thresholds[0] = -np.inf
    for i in range(4):
        mask = (x > thresholds[i])
        xi = x[mask]
        labels = (y[mask] > i)
        tr = find_best_threshold(xi, labels, tr0=thresholds[i])
        thresholds[i+1] = tr
    return thresholds


def calculate_metrics(scores, y, thresholds):
    ## calculate accuracy, r, rho
    pred_level = np.digitize(scores, thresholds[1:], right=True)
    accuracy = np.mean(pred_level == y)
    r,_ = pearsonr(scores, y)
    rho,_ = spearmanr(scores, y)
    ## calculate average-precision score for each structural level
    aupr = np.zeros(4, dtype=np.float32)
    for i in range(4):
        target = (y > i).astype(np.float32)
        aupr[i] = average_precision(target, scores.astype(np.float32))
    return accuracy, r, rho, aupr


def score_pairs(model, x0, x1, batch_size=100):
    scores = []
    for i in range(0, len(x0), batch_size):
        x0_mb = x0[i:i+batch_size]
        x1_mb = x1[i:i+batch_size]
        scores.append(model(x0_mb, x1_mb))
    scores = np.concatenate(scores, 0)
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser('Script for evaluating similarity model on SCOP test set.')

    parser.add_argument('model', help='path to saved model file or "nw-align" for Needleman-Wunsch alignment score baseline')

    parser.add_argument('--dev', action='store_true', help='use train/dev split')

    parser.add_argument('--batch-size', default=64, type=int, help='number of sequence pairs to process in each batch (default: 64)')

    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')

    parser.add_argument('--coarse', action='store_true', help='use coarse comparison rather than full SSA')

    args = parser.parse_args()

    scop_train_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.sampledpairs.txt'

    eval_paths = [ ( '2.06-test'
                   , 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.sampledpairs.txt')
                 , ( '2.07-new'
                   , 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.07-new.allpairs.txt')
                 ] 
    if args.dev:
        scop_train_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.train.sampledpairs.txt'

        eval_paths = [ ( '2.06-dev'
                       , 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.dev.sampledpairs.txt')
                     ] 


    ## load the data
    alphabet = Uniprot21()
    x0_train, x1_train, y_train = load_pairs(scop_train_path, alphabet)

    ## load the model
    if args.model == 'nw-align':
        model = NWAlign(alphabet)
    elif args.model in ['hhalign', 'phmmer', 'TMalign']:
        model = args.model
    else:
        model = torch.load(args.model)
        model.eval()

        ## set the device
        d = args.device
        use_cuda = (d != -1) and torch.cuda.is_available()
        if d >= 0:
            torch.cuda.set_device(d)

        if use_cuda:
            model.cuda()

        mode = 'align'
        if args.coarse:
            mode = 'coarse'
        model = TorchModel(model, use_cuda, mode=mode)

    batch_size = args.batch_size

    ## for calculating the classification accuracy, first find the best partitions using the training set
    if type(model) is str:
        path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.sampledpairs.' \
                + model + '.npy'
        scores = np.load(path)
        scores = scores.mean(1)
    else:
        scores = score_pairs(model, x0_train, x1_train, batch_size)
    thresholds = find_best_thresholds(scores, y_train)

    print('Dataset\tAccuracy\tPearson\'s r\tSpearman\'s rho\tClass\tFold\tSuperfamily\tFamily')

    accuracy, r, rho, aupr = calculate_metrics(scores, y_train, thresholds)

    template = '{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'

    line = '2.06-train\t' + template.format(accuracy, r, rho, aupr[0], aupr[1], aupr[2], aupr[3])
    #line = '\t'.join(['2.06-train', str(accuracy), str(r), str(rho), str(aupr[0]), str(aupr[1]), str(aupr[2]), str(aupr[3])])
    print(line)

    for dset,path in eval_paths:
        x0_test, x1_test, y_test = load_pairs(path, alphabet)
        if type(model) is str:
            path = os.path.splitext(path)[0]
            path = path + '.' + model + '.npy'
            scores = np.load(path)
            scores = scores.mean(1)
        else:
            scores = score_pairs(model, x0_test, x1_test, batch_size)
        accuracy, r, rho, aupr = calculate_metrics(scores, y_test, thresholds)

        line = dset + '\t' + template.format(accuracy, r, rho, aupr[0], aupr[1], aupr[2], aupr[3])
        #line = '\t'.join([dset, str(accuracy), str(r), str(rho), str(aupr[0]), str(aupr[1]), str(aupr[2]), str(aupr[3])])
        print(line)


if __name__ == '__main__':
    main()

