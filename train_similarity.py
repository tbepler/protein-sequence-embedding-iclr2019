from __future__ import print_function,division

import numpy as np
import pandas as pd
import sys

from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.utils.data

from src.alphabets import Uniprot21
import src.scop as scop
from src.utils import pack_sequences, unpack_sequences
from src.utils import PairedDataset, AllPairsDataset, collate_paired_sequences
from src.utils import MultinomialResample
import src.models.embedding
import src.models.comparison


def main():
    import argparse
    parser = argparse.ArgumentParser('Script for training embedding model on SCOP.')

    parser.add_argument('--dev', action='store_true', help='use train/dev split')

    parser.add_argument('-m', '--model', choices=['ssa', 'ua', 'me'], default='ssa', help='alignment scoring method for comparing sequences in embedding space [ssa: soft symmetric alignment, ua: uniform alignment, me: mean embedding] (default: ssa)')
    parser.add_argument('--allow-insert', action='store_true', help='model insertions (default: false)')

    parser.add_argument('--norm', choices=['l1', 'l2'], default='l1', help='comparison norm (default: l1)')

    parser.add_argument('--rnn-type', choices=['lstm', 'gru'], default='lstm', help='type of RNN block to use (default: lstm)')
    parser.add_argument('--embedding-dim', type=int, default=100, help='embedding dimension (default: 100)')
    parser.add_argument('--input-dim', type=int, default=512, help='dimension of input to RNN (default: 512)')
    parser.add_argument('--rnn-dim', type=int, default=512, help='hidden units of RNNs (default: 512)')
    parser.add_argument('--num-layers', type=int, default=3, help='number of RNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability (default: 0)')

    parser.add_argument('--epoch-size', type=int, default=100000, help='number of examples per epoch (default: 100,000)')
    parser.add_argument('--epoch-scale', type=int, default=5, help='scaling on epoch size (default: 5)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs (default: 100)')

    parser.add_argument('--batch-size', type=int, default=64, help='minibatch size (default: 64)')

    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--tau', type=float, default=0.5, help='sampling proportion exponent (default: 0.5)')
    parser.add_argument('--augment', type=float, default=0, help='probability of resampling amino acid for data augmentation (default: 0)')
    parser.add_argument('--lm', help='pretrained LM to use as initial embedding')

    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')

    args = parser.parse_args()


    prefix = args.output


    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    ## make the datasets
    astral_train_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.fa'
    astral_testpairs_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.sampledpairs.txt'
    if args.dev:
        astral_train_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.train.fa'
        astral_testpairs_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.dev.sampledpairs.txt'

    alphabet = Uniprot21()

    print('# loading training sequences:', astral_train_path, file=sys.stderr)
    with open(astral_train_path, 'rb') as f:
        names_train, structs_train, sequences_train = scop.parse_astral(f, encoder=alphabet)    
    x_train = [torch.from_numpy(x).long() for x in sequences_train]
    if use_cuda:
        x_train = [x.cuda() for x in x_train]
    y_train = torch.from_numpy(structs_train)

    print('# loaded', len(x_train), 'training sequences', file=sys.stderr)


    print('# loading test sequence pairs:', astral_testpairs_path, file=sys.stderr)
    test_pairs_table = pd.read_csv(astral_testpairs_path, sep='\t') 
    x0_test = [x.encode('utf-8').upper() for x in test_pairs_table['sequence_A']]
    x0_test = [torch.from_numpy(alphabet.encode(x)).long() for x in x0_test]
    x1_test = [x.encode('utf-8').upper() for x in test_pairs_table['sequence_B']]
    x1_test = [torch.from_numpy(alphabet.encode(x)).long() for x in x1_test]
    if use_cuda:
        x0_test = [x.cuda() for x in x0_test]
        x1_test = [x.cuda() for x in x1_test]
    y_test = test_pairs_table['similarity'].values
    y_test = torch.from_numpy(y_test).long()

    dataset_test = PairedDataset(x0_test, x1_test, y_test)
    print('# loaded', len(x0_test), 'test pairs', file=sys.stderr)

    ## make the dataset iterators
    scale = args.epoch_scale

    epoch_size = args.epoch_size
    batch_size = args.batch_size

    # precompute the similarity pairs
    y_train_levels = torch.cumprod((y_train.unsqueeze(1) == y_train.unsqueeze(0)).long(), 2)

    # data augmentation by resampling amino acids
    augment = None
    p = 0
    if args.augment > 0:
        p = args.augment
        trans = torch.ones(len(alphabet),len(alphabet))
        trans = trans/trans.sum(1, keepdim=True)
        if use_cuda:
            trans = trans.cuda()
        augment = MultinomialResample(trans, p)
    print('# resampling amino acids with p:', p, file=sys.stderr)
    dataset_train = AllPairsDataset(x_train, y_train_levels, augment=augment)

    similarity = y_train_levels.numpy().sum(2)
    levels,counts = np.unique(similarity, return_counts=True)
    order = np.argsort(levels)
    levels = levels[order]
    counts = counts[order]

    print('#', levels, file=sys.stderr)
    print('#', counts/np.sum(counts), file=sys.stderr)

    weight = counts**0.5
    print('#', weight/np.sum(weight), file=sys.stderr)

    weight = counts**0.33
    print('#', weight/np.sum(weight), file=sys.stderr)

    weight = counts**0.25
    print('#', weight/np.sum(weight), file=sys.stderr)

    tau = args.tau
    print('# using tau:', tau, file=sys.stderr)
    print('#', counts**tau/np.sum(counts**tau), file=sys.stderr)
    weights = counts**tau/counts
    weights = weights[similarity].ravel()
    #weights = np.ones(len(dataset_train))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, epoch_size)

    # two training dataset iterators for sampling pairs of sequences for training
    train_iterator = torch.utils.data.DataLoader(dataset_train
                                                , batch_size=batch_size
                                                , sampler=sampler
                                                , collate_fn=collate_paired_sequences
                                                )
    test_iterator = torch.utils.data.DataLoader(dataset_test
                                               , batch_size=batch_size
                                               , collate_fn=collate_paired_sequences
                                               )
    

    ## initialize the model 
    rnn_type = args.rnn_type
    rnn_dim = args.rnn_dim
    num_layers = args.num_layers

    embedding_size = args.embedding_dim
    input_dim = args.input_dim

    dropout = args.dropout
    
    allow_insert = args.allow_insert

    print('# initializing model with:', file=sys.stderr)
    print('# embedding_size:', embedding_size, file=sys.stderr)
    print('# input_dim:', input_dim, file=sys.stderr)
    print('# rnn_dim:', rnn_dim, file=sys.stderr)
    print('# num_layers:', num_layers, file=sys.stderr)
    print('# dropout:', dropout, file=sys.stderr)
    print('# allow_insert:', allow_insert, file=sys.stderr)

    compare_type = args.model
    print('# comparison method:', compare_type, file=sys.stderr)

    lm = None
    if args.lm is not None:
        lm = torch.load(args.lm)
        lm.eval()
        ## do not update the LM parameters
        for param in lm.parameters():
            param.requires_grad = False
        print('# using LM:', args.lm, file=sys.stderr)

    if num_layers > 0:
        embedding = src.models.embedding.StackedRNN(len(alphabet), input_dim, rnn_dim, embedding_size
                                                   , nlayers=num_layers, dropout=dropout, lm=lm)
    else:
        embedding = src.models.embedding.Linear(len(alphabet), input_dim, embedding_size, lm=lm)

    if args.norm == 'l1':
        norm = src.models.comparison.L1()
        print('# norm: l1', file=sys.stderr)
    elif args.norm == 'l2':
        norm = src.models.comparison.L2()
        print('# norm: l2', file=sys.stderr)
    model = src.models.comparison.OrdinalRegression(embedding, 5, align_method=compare_type
                                                   , compare=norm, allow_insertions=allow_insert
                                                   )

    if use_cuda:
        model.cuda()

    ## setup training parameters and optimizer
    num_epochs = args.num_epochs

    weight_decay = args.weight_decay
    lr = args.lr

    print('# training with Adam: lr={}, weight_decay={}'.format(lr, weight_decay), file=sys.stderr)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    ## train the model
    print('# training model', file=sys.stderr)

    save_prefix = args.save_prefix
    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')
    digits = int(np.floor(np.log10(num_epochs))) + 1
    line = '\t'.join(['epoch', 'split', 'loss', 'mse', 'accuracy', 'r', 'rho' ])
    print(line, file=output)


    for epoch in range(num_epochs):
        # train epoch
        model.train()
        it = 0
        n = 0
        loss_estimate = 0
        mse_estimate = 0
        acc_estimate = 0

        for x0,x1,y in train_iterator: # zip(train_iterator_0, train_iterator_1):

            if use_cuda:
                y = y.cuda()
            y = Variable(y)

            b = len(x0)
            x = x0 + x1

            x,order = pack_sequences(x)
            x = PackedSequence(Variable(x.data), x.batch_sizes)
            z = model(x) # embed the sequences
            z = unpack_sequences(z, order)

            z0 = z[:b]
            z1 = z[b:]

            logits = []
            for i in range(b):
                z_a = z0[i]
                z_b = z1[i]
                logits.append(model.score(z_a, z_b))
            logits = torch.stack(logits, 0)

            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            loss.backward()

            optim.step()
            optim.zero_grad()
            model.clip() # projected gradient for bounding ordinal regressionn parameters

            p = F.sigmoid(logits) 
            ones = p.new(b,1).zero_() + 1
            p_ge = torch.cat([ones, p], 1)
            p_lt = torch.cat([1-p, ones], 1)
            p = p_ge*p_lt
            p = p/p.sum(1,keepdim=True) # make sure p is normalized

            _,y_hard = torch.max(p, 1)
            levels = torch.arange(5).to(p.device)
            y_hat = torch.sum(p*levels, 1)
            y = torch.sum(y.data, 1)

            loss = F.cross_entropy(p, y) # calculate cross entropy loss from p vector

            correct = torch.sum((y == y_hard).float())
            mse = torch.sum((y.float() - y_hat)**2)

            n += b
            delta = b*(loss.item() - loss_estimate)
            loss_estimate += delta/n
            delta = correct.item() - b*acc_estimate
            acc_estimate += delta/n
            delta = mse.item() - b*mse_estimate
            mse_estimate += delta/n

            
            if (n - b)//100 < n//100:
                print('# [{}/{}] training {:.1%} loss={:.5f}, mse={:.5f}, acc={:.5f}'.format(epoch+1
                                                                , num_epochs
                                                                , n/epoch_size
                                                                , loss_estimate
                                                                , mse_estimate 
                                                                , acc_estimate 
                                                                )
                     , end='\r', file=sys.stderr)
        print(' '*80, end='\r', file=sys.stderr)
        line = '\t'.join([str(epoch+1).zfill(digits), 'train', str(loss_estimate)
                         , str(mse_estimate), str(acc_estimate), '-', '-'])
        print(line, file=output)
        output.flush()

        # eval and save model
        model.eval()

        y = []
        logits = []
        with torch.no_grad():
            for x0,x1,y_mb in test_iterator:

                if use_cuda:
                    y_mb = y_mb.cuda()
                y.append(y_mb.long())

                b = len(x0)
                x = x0 + x1

                x,order = pack_sequences(x)
                x = PackedSequence(Variable(x.data), x.batch_sizes)
                z = model(x) # embed the sequences
                z = unpack_sequences(z, order)

                z0 = z[:b]
                z1 = z[b:]

                for i in range(b):
                    z_a = z0[i]
                    z_b = z1[i]
                    logits.append(model.score(z_a, z_b))

            y = torch.cat(y, 0)
            logits = torch.stack(logits, 0)

            p = F.sigmoid(logits).data 
            ones = p.new(p.size(0),1).zero_() + 1
            p_ge = torch.cat([ones, p], 1)
            p_lt = torch.cat([1-p, ones], 1)
            p = p_ge*p_lt
            p = p/p.sum(1,keepdim=True) # make sure p is normalized

            loss = F.cross_entropy(p, y).item()

            _,y_hard = torch.max(p, 1)
            levels = torch.arange(5).to(p.device)
            y_hat = torch.sum(p*levels, 1)

            accuracy = torch.mean((y == y_hard).float()).item()
            mse = torch.mean((y.float() - y_hat)**2).item()

            y = y.cpu().numpy()
            y_hat = y_hat.cpu().numpy()

            r,_ = pearsonr(y_hat, y)
            rho,_ = spearmanr(y_hat, y)

        line = '\t'.join([str(epoch+1).zfill(digits), 'test', str(loss), str(mse)
                         , str(accuracy), str(r), str(rho)])
        print(line, file=output)
        output.flush()


        # save the model
        if save_prefix is not None:
            save_path = save_prefix + '_epoch' + str(epoch+1).zfill(digits) + '.sav'
            model.cpu()
            torch.save(model, save_path)
            if use_cuda:
                model.cuda()




if __name__ == '__main__':
    main()





