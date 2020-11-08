from __future__ import print_function, division

import numpy as np
import sys
import os
import glob
from PIL import Image

import torch
from torch.nn.utils.rnn import PackedSequence
import torch.utils.data

from bepler.alphabets import Uniprot21
import bepler.fasta as fasta
from bepler.utils import pack_sequences, unpack_sequences
from bepler.utils import ContactMapDataset, collate_lists
from bepler.metrics import average_precision


def load_data(seq_path, struct_path, alphabet, baselines=False):

    pdb_index = {}
    for path in struct_path:
        pid = os.path.basename(path)[:7]
        pdb_index[pid] = path

    with open(seq_path, 'rb') as f:
        names, sequences = fasta.parse(f)
    names = [name.split()[0].decode('utf-8') for name in names]
    sequences = [alphabet.encode(s.upper()) for s in sequences]

    x = [torch.from_numpy(x).long() for x in sequences]

    names_ = []
    x_ = []
    y = []
    for xi,name in zip(x,names):
        pid = name
        if pid not in pdb_index:
            pid = 'd' + pid[1:]
        path = pdb_index[pid]

        im = np.array(Image.open(path), copy=False)
        contacts = np.zeros(im.shape, dtype=np.float32)
        contacts[im == 1] = -1
        contacts[im == 255] = 1

        # mask the matrix below the diagonal
        mask = np.tril_indices(contacts.shape[0], k=1)
        contacts[mask] = -1

        names_.append(name)
        x_.append(xi)
        y.append(torch.from_numpy(contacts))

    return x_, y, names_


def predict_minibatch(model, x, use_cuda):
    b = len(x)
    x,order = pack_sequences(x)
    x = PackedSequence(x.data, x.batch_sizes)
    z = model(x) # embed the sequences
    z = unpack_sequences(z, order)

    logits = []
    for i in range(b):
        zi = z[i]
        lp = model.predict(zi.unsqueeze(0)).view(zi.size(0), zi.size(0))
        logits.append(lp)

    return logits


def calc_metrics(logits, y):
    y_hat = (logits > 0).astype(np.float32)
    TP = (y_hat*y).sum()
    precision = 1.0
    if y_hat.sum() > 0:
        precision = TP/y_hat.sum()
    recall = TP/y.sum()
    F1 = 0
    if precision + recall > 0:
        F1 = 2*precision*recall/(precision + recall)
    AUPR = average_precision(y, logits)
    return precision, recall, F1, AUPR


def main():
    import argparse
    parser = argparse.ArgumentParser('Script for evaluating contact map models.')
    parser.add_argument('model', help='path to saved model')
    parser.add_argument('--dataset', default='2.06 test', help='which dataset (default: 2.06 test)')
    parser.add_argument('--batch-size', default=10, type=int, help='number of sequences to process in each batch (default: 10)')
    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    args = parser.parse_args()

    # load the data
    if args.dataset == '2.06 test':
        fasta_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.fa'
        contact_paths = glob.glob('data/SCOPe/pdbstyle-2.06/*/*.png')
    elif args.dataset == '2.07 test' or args.dataset == '2.07 new test':
        fasta_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.07-new.fa'
        contact_paths = glob.glob('data/SCOPe/pdbstyle-2.07/*/*.png')
    else:
        raise Exception('Bad dataset argument ' + args.dataset)
    
    alphabet = Uniprot21()
    x,y,names = load_data(fasta_path, contact_paths, alphabet)

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    if use_cuda:
        x = [x_.cuda() for x_ in x]
        y = [y_.cuda() for y_ in y]

    model = torch.load(args.model)
    model.eval()
    if use_cuda:
        model.cuda()

    # predict contact maps
    batch_size = args.batch_size
    dataset = ContactMapDataset(x, y)
    iterator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_lists)
    logits = []
    with torch.no_grad():
        for xmb,ymb in iterator:
            lmb = predict_minibatch(model, xmb, use_cuda)
            logits += lmb

    # calculate performance metrics
    lengths = np.array([len(x_) for x_ in x])
    logits = [logit.cpu().numpy() for logit in logits]
    y = [y_.cpu().numpy() for y_ in y]

    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')
    line = '\t'.join(['Distance', 'Precision', 'Recall', 'F1', 'AUPR', 'Precision@L', 'Precision@L/2', 'Precision@L/5'])
    print(line, file=output)
    output.flush()

    # for all contacts
    y_flat = []
    logits_flat = []
    for i in range(len(y)):
        yi = y[i]
        mask = (yi < 0)
        y_flat.append(yi[~mask])
        logits_flat.append(logits[i][~mask])

    # calculate precision, recall, F1, and area under the precision recall curve for all contacts
    precision = np.zeros(len(x))
    recall = np.zeros(len(x))
    F1 = np.zeros(len(x))
    AUPR = np.zeros(len(x))
    prL = np.zeros(len(x))
    prL2 = np.zeros(len(x))
    prL5 = np.zeros(len(x))
    for i in range(len(x)):
        pr,re,f1,aupr = calc_metrics(logits_flat[i], y_flat[i])
        precision[i] = pr
        recall[i] = re
        F1[i] = f1
        AUPR[i] = aupr

        order = np.argsort(logits_flat[i])[::-1]
        n = lengths[i]
        topL = order[:n]
        prL[i] = y_flat[i][topL].mean()
        topL2 = order[:n//2]
        prL2[i] = y_flat[i][topL2].mean()
        topL5 = order[:n//5]
        prL5[i] = y_flat[i][topL5].mean()

    template = 'All\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'
    line = template.format(precision.mean(), recall.mean(), F1.mean(), AUPR.mean(), prL.mean(), prL2.mean(), prL5.mean())
    print(line, file=output)
    output.flush()

    # for Medium/Long range contacts
    y_flat = []
    logits_flat = []
    for i in range(len(y)):
        yi = y[i]
        mask = (yi < 0)

        medlong = np.tril_indices(len(yi), k=11)
        medlong_mask = np.zeros((len(yi),len(yi)), dtype=np.uint8)
        medlong_mask[medlong] = 1
        mask = mask | (medlong_mask == 1)

        y_flat.append(yi[~mask])
        logits_flat.append(logits[i][~mask])

    # calculate precision, recall, F1, and area under the precision recall curve for all contacts
    precision = np.zeros(len(x))
    recall = np.zeros(len(x))
    F1 = np.zeros(len(x))
    AUPR = np.zeros(len(x))
    prL = np.zeros(len(x))
    prL2 = np.zeros(len(x))
    prL5 = np.zeros(len(x))
    for i in range(len(x)):
        pr,re,f1,aupr = calc_metrics(logits_flat[i], y_flat[i])
        precision[i] = pr
        recall[i] = re
        F1[i] = f1
        AUPR[i] = aupr

        order = np.argsort(logits_flat[i])[::-1]
        n = lengths[i]
        topL = order[:n]
        prL[i] = y_flat[i][topL].mean()
        topL2 = order[:n//2]
        prL2[i] = y_flat[i][topL2].mean()
        topL5 = order[:n//5]
        prL5[i] = y_flat[i][topL5].mean()

    template = 'Medium/Long\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'
    line = template.format(np.nanmean(precision), np.nanmean(recall), np.nanmean(F1), np.nanmean(AUPR), np.nanmean(prL)
                          , np.nanmean(prL2), np.nanmean(prL5))
    print(line, file=output)
    output.flush()



if __name__ == '__main__':
    main()





