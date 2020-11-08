from __future__ import print_function,division

import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence

import bepler.fasta as fasta
from bepler.alphabets import Uniprot21
import bepler.models.sequence

parser = argparse.ArgumentParser('Train sequence model')

parser.add_argument('-b', '--minibatch-size', type=int, default=32, help='minibatch size (default: 32)')
parser.add_argument('-n', '--num-epochs', type=int, default=10, help='number of epochs (default: 10)')

parser.add_argument('--hidden-dim', type=int, default=512, help='hidden dimension of RNN (default: 512)')
parser.add_argument('--num-layers', type=int, default=2, help='number of RNN layers (default: 2)')
parser.add_argument('--dropout', type=float, default=0, help='dropout (default: 0)')

parser.add_argument('--untied', action='store_true', help='use biRNN with untied weights')

parser.add_argument('--l2', type=float, default=0, help='l2 regularizer (default: 0)')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping max norm (default: 1)')

parser.add_argument('-d', '--device', type=int, default=-2, help='device to use, -1: cpu, 0+: gpu (default: gpu if available, else cpu)')

parser.add_argument('-o', '--output', help='where to write training curve (default: stdout)')
parser.add_argument('--save-prefix', help='path prefix for saving models (default: no saving)')


pfam_train = 'data/pfam/Pfam-A.train.fasta'
pfam_test = 'data/pfam/Pfam-A.test.fasta'


def preprocess_sequence(s, alphabet):
    x = alphabet.encode(s)
    # pad with start/stop token
    z = np.zeros(len(x)+2, dtype=x.dtype)
    z[1:-1] = x + 1
    return z

def load_pfam(path, alph):
    # load path sequences and families
    with open(path, 'rb') as f:
        group = []
        sequences = []
        for name,sequence in fasta.parse_stream(f):
            x = preprocess_sequence(sequence.upper(), alph)
            sequences.append(x)
            family = name.split(b';')[-2]
            group.append(family)
    group = np.array(group)
    sequences = np.array(sequences)
    return group, sequences


def main():
    args = parser.parse_args()

    alph = Uniprot21()
    ntokens = len(alph)

    ## load the training sequences
    train_group, X_train = load_pfam(pfam_train, alph)
    print('# loaded', len(X_train), 'sequences from', pfam_train, file=sys.stderr)

    ## load the testing sequences
    test_group, X_test = load_pfam(pfam_test, alph)
    print('# loaded', len(X_test), 'sequences from', pfam_test, file=sys.stderr)

    ## initialize the model
    nin = ntokens + 1
    nout = ntokens
    embedding_dim = 21
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    mask_idx = ntokens
    dropout = args.dropout

    tied = not args.untied

    model = bepler.models.sequence.BiLM(nin, nout, embedding_dim, hidden_dim, num_layers
                                        , mask_idx=mask_idx, dropout=dropout, tied=tied)
    print('# initialized model', file=sys.stderr)

    device = args.device
    use_cuda = torch.cuda.is_available() and (device == -2 or device >= 0)
    if device >= 0:
        torch.cuda.set_device(device)
    if use_cuda:
        model = model.cuda()

    ## form the data iterators and optimizer
    lr = args.lr
    l2 = args.l2
    solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    def collate(xs):
        B = len(xs)
        N = max(len(x) for x in xs)
        lengths = np.array([len(x) for x in xs], dtype=int)
        
        order = np.argsort(lengths)[::-1]
        lengths = lengths[order]

        X = torch.LongTensor(B, N).zero_() + mask_idx
        for i in range(B):
            x = xs[order[i]]
            n = len(x)
            X[i,:n] = torch.from_numpy(x)
        return X, lengths

    mb = args.minibatch_size

    train_iterator = torch.utils.data.DataLoader(X_train, batch_size=mb, shuffle=True
                                                , collate_fn=collate)
    test_iterator = torch.utils.data.DataLoader(X_test, batch_size=mb
                                               , collate_fn=collate)

    ## fit the model!

    print('# training model', file=sys.stderr)

    output = sys.stdout
    if args.output is not None:
        output = open(args.output, 'w')

    num_epochs = args.num_epochs
    clip = args.clip

    save_prefix = args.save_prefix
    digits = int(np.floor(np.log10(num_epochs))) + 1

    print('epoch\tsplit\tlog_p\tperplexity\taccuracy', file=output)
    output.flush()

    for epoch in range(num_epochs):
        # train epoch
        model.train()
        it = 0
        n = 0
        accuracy = 0
        loss_accum = 0
        for X,lengths in train_iterator:
            if use_cuda:
                X = X.cuda()
            X = Variable(X)
            logp = model(X)

            mask = (X != mask_idx)

            index = X*mask.long()
            loss = -logp.gather(2, index.unsqueeze(2)).squeeze(2)
            loss = torch.mean(loss.masked_select(mask))

            loss.backward()

            # clip the gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            solver.step()
            solver.zero_grad()

            _,y_hat = torch.max(logp, 2)
            correct = torch.sum((y_hat == X).masked_select(mask))
            #correct = torch.sum((y_hat == X)[mask.nonzero()].float())

            b = mask.long().sum().item()
            n += b
            delta = b*(loss.item() - loss_accum)
            loss_accum += delta/n
            delta = correct.item() - b*accuracy
            accuracy += delta/n

            b = X.size(0)
            it += b
            if (it - b)//100 < it//100:
                print('# [{}/{}] training {:.1%} loss={:.5f}, acc={:.5f}'.format(epoch+1
                                                                , num_epochs
                                                                , it/len(X_train)
                                                                , loss_accum
                                                                , accuracy
                                                                )
                     , end='\r', file=sys.stderr)
        print(' '*80, end='\r', file=sys.stderr)

        perplex = np.exp(loss_accum)
        string = str(epoch+1).zfill(digits) + '\t' + 'train' + '\t' + str(loss_accum) \
                 + '\t' + str(perplex) + '\t' + str(accuracy)
        print(string, file=output)
        output.flush()

        # test epoch
        model.eval()
        it = 0
        n = 0
        accuracy = 0
        loss_accum = 0
        with torch.no_grad():
            for X,lengths in test_iterator:
                if use_cuda:
                    X = X.cuda()
                X = Variable(X)
                logp = model(X)

                mask = (X != mask_idx)

                index = X*mask.long()
                loss = -logp.gather(2, index.unsqueeze(2)).squeeze(2)
                loss = torch.mean(loss.masked_select(mask))

                _,y_hat = torch.max(logp, 2)
                correct = torch.sum((y_hat == X).masked_select(mask))

                b = mask.long().sum().item()
                n += b
                delta = b*(loss.item() - loss_accum)
                loss_accum += delta/n
                delta = correct.item() - b*accuracy
                accuracy += delta/n

                b = X.size(0)
                it += b
                if (it - b)//100 < it//100:
                    print('# [{}/{}] test {:.1%} loss={:.5f}, acc={:.5f}'.format(epoch+1
                                                                    , num_epochs
                                                                    , it/len(X_test)
                                                                    , loss_accum
                                                                    , accuracy
                                                                    )
                         , end='\r', file=sys.stderr)
        print(' '*80, end='\r', file=sys.stderr)

        perplex = np.exp(loss_accum)
        string = str(epoch+1).zfill(digits) + '\t' + 'test' + '\t' + str(loss_accum) \
                 + '\t' + str(perplex) + '\t' + str(accuracy)
        print(string, file=output)
        output.flush()

        ## save the model
        if save_prefix is not None:
            save_path = save_prefix + '_epoch' + str(epoch+1).zfill(digits) + '.sav'
            model = model.cpu()
            torch.save(model, save_path)
            if use_cuda:
                model = model.cuda()



if __name__ == '__main__':
    main()











