from __future__ import print_function,division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.utils.data

from src.alphabets import Uniprot21, SecStr8
from src.utils import pack_sequences, unpack_sequences
import src.pdb as pdb


secstr_train_path = 'data/secstr/ss_cullpdb_pc40_res3.0_R1.0_d180412_filtered.train.fa'
secstr_test_path = 'data/secstr/ss_cullpdb_pc40_res3.0_R1.0_d180412_filtered.test.fa'


def encode_sequence(x, alphabet):
    # convert to bytes and uppercase
    x = x.encode('utf-8').upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    return x


def load_secstr(path, alphabet, secstr):
    with open(path, 'rb') as f:
        names,aa_seqs,ss_seqs = pdb.parse_secstr(f)
    aa_seqs = [alphabet.encode(x.upper()) for x in aa_seqs]
    ss_seqs = [secstr.encode(x.upper()) for x in ss_seqs]
    return names,aa_seqs,ss_seqs


def unstack_lstm(lstm):
    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)
            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
        layers.append(layer)
        in_size = 2*hidden_dim
    return layers

def featurize(x, lm_embed, lstm_stack, proj, include_lm=True, lm_only=False):
    zs = []

    packed = type(x) is PackedSequence
    if packed:
        batch_sizes = x.batch_sizes
        x = x.data

    x_onehot = x.new(x.size(0), 21).float().zero_()
    x_onehot.scatter_(1,x.unsqueeze(1),1)

    if packed:
        x_onehot = PackedSequence(x_onehot, batch_sizes)
        x = PackedSequence(x, batch_sizes)

    zs.append(x_onehot)
    
    h = lm_embed(x)
    if include_lm:
        zs.append(h)
    if not lm_only:
        for lstm in lstm_stack:
            h,_ = lstm(h)
            zs.append(h)
        if packed:
            h = h.data
        h = proj(h)
        if packed:
            h = PackedSequence(h, batch_sizes)
        zs.append(h)
    if packed:
        zs = [z.data for z in zs]
    z = torch.cat(zs, 1)
    if packed:
        z = PackedSequence(z, batch_sizes)
    return z

class TorchModel:
    def __init__(self, model, use_cuda, full_features=False):
        self.model = model
        self.use_cuda = use_cuda
        self.full_features = full_features
        if full_features:
            self.lm_embed = model.embedding.embed
            self.lstm_stack = unstack_lstm(model.embedding.rnn)
            self.proj = model.embedding.proj
            if use_cuda:
                self.lm_embed.cuda()
                for lstm in self.lstm_stack:
                    lstm.cuda()
                self.proj.cuda()


    def __call__(self, x):
        c = [torch.from_numpy(x_).long() for x_ in x]

        c,order = pack_sequences(c)
        if self.use_cuda:
            c = c.cuda()

        if self.full_features:
            z = featurize(c, self.lm_embed, self.lstm_stack, self.proj)
        else:
            z = self.model(c) # embed the sequences
        z = unpack_sequences(z, order)

        return z

def kmer_features(xs, n, k):
    if k == 1:
        return xs, n
    pad = np.array([n]*(k//2))
    f = (n+1)**np.arange(k)
    kmers = []
    for x in xs:
        x = np.concatenate([pad, x, pad], axis=0)
        z = np.convolve(x, f, mode='valid')
        kmers.append(z)
    return kmers, (n+1)**k


class Shuffle:
    def __init__(self, x, y, minibatch_size):
        self.x = x
        self.y = y
        self.minibatch_size = minibatch_size

    def __iter__(self):
        n = len(self.x)
        order = np.random.permutation(n)
        order = torch.from_numpy(order).to(self.x.device)
        x = self.x[order]
        y = self.y[order]
        b = self.minibatch_size
        for i in range(0, n, b):
            yield x[i:i+b], y[i:i+b]


def fit_kmer_potentials(x, y, n, m):
    _,counts = np.unique(y, return_counts=True)
    weights = torch.zeros(n, m)
    weights += torch.from_numpy(counts/counts.sum()).float()
    for i in range(len(x)):
        weights[x[i],y[i]] += 1

    model = nn.Embedding(n, m, sparse=True)
    model.weight.data[:] = torch.log(weights) - torch.log(weights.sum(1, keepdim=True))

    return model


def fit_nn_potentials(model, x, y, lr=0.001, num_epochs=10, minibatch_size=256
                     , use_cuda=False):
    solver = torch.optim.Adam(model.parameters(), lr=lr)

    iterator = Shuffle(x, y, minibatch_size)

    model.train()
    for epoch in range(num_epochs):
        n = 0
        loss_accum = 0
        acc = 0
        for x,y in iterator:
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            potentials = model(x).view(x.size(0), -1)
            loss = F.cross_entropy(potentials, y)

            loss.backward()
            solver.step()
            solver.zero_grad()

            _,y_hat = potentials.max(1)
            correct = torch.sum((y_hat == y).float())

            b = x.size(0)
            n += b
            delta = b*(loss.item() - loss_accum)
            loss_accum += delta/n
            delta = correct.item() - b*acc
            acc += delta/n

        print('train', epoch+1, loss_accum, np.exp(loss_accum), acc)


def main():
    import argparse
    parser = argparse.ArgumentParser('Script for evaluating similarity model on SCOP test set.')

    parser.add_argument('features', help='path to saved embedding model file or "1-", "3-", or "5-mer" for k-mer features')

    parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs to train for (default: 10)')
    parser.add_argument('--all-hidden', action='store_true', help='use all hidden layers as features')

    parser.add_argument('-v', '--print-examples', default=0, type=int, help='number of examples to print (default: 0)')

    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')

    args = parser.parse_args()
    num_epochs = args.num_epochs

    ## load the data
    alphabet = Uniprot21()
    secstr = SecStr8

    names_train, x_train, y_train = load_secstr(secstr_train_path, alphabet, secstr)
    names_test, x_test, y_test = load_secstr(secstr_test_path, alphabet, secstr)

    sequences_test = [''.join(alphabet[c] for c in x_test[i]) for i in range(len(x_test))]

    y_train = np.concatenate(y_train, 0)

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)


    if args.features == '1-mer':
        n = len(alphabet)
        x_test = [x.astype(int) for x in x_test]
    elif args.features == '3-mer':
        x_train,n = kmer_features(x_train, len(alphabet), 3)
        x_test,_ = kmer_features(x_test, len(alphabet), 3)
    elif args.features == '5-mer':
        x_train,n = kmer_features(x_train, len(alphabet), 5)
        x_test,_ = kmer_features(x_test, len(alphabet), 5)
    else:
        features = torch.load(args.features)
        features.eval()

        if use_cuda:
            features.cuda()

        features = TorchModel(features, use_cuda, full_features=args.all_hidden)
        batch_size = 32 # batch size for featurizing sequences

        with torch.no_grad():
            z_train = []
            for i in range(0,len(x_train),batch_size):
                for z in features(x_train[i:i+batch_size]):
                    z_train.append(z.cpu().numpy())
            x_train = z_train

            z_test = []
            for i in range(0,len(x_test),batch_size):
                for z in features(x_test[i:i+batch_size]):
                    z_test.append(z.cpu().numpy())
            x_test = z_test

        n = x_train[0].shape[1]
        del features
        del z_train
        del z_test

    print('split', 'epoch', 'loss', 'perplexity', 'accuracy')

    if args.features.endswith('-mer'):
        x_train = np.concatenate(x_train, 0)
        model = fit_kmer_potentials(x_train, y_train, n, len(secstr))
    else:
        x_train = torch.cat([torch.from_numpy(x) for x in x_train], 0)
        if use_cuda and not args.all_hidden:
            x_train = x_train.cuda()

        num_hidden = 1024
        model = nn.Sequential( nn.Linear(n, num_hidden)
                             , nn.ReLU()
                             , nn.Linear(num_hidden, num_hidden)
                             , nn.ReLU()
                             , nn.Linear(num_hidden, len(secstr))
                             )
            
        y_train = torch.from_numpy(y_train).long()
        if use_cuda:
            y_train = y_train.cuda()
            model.cuda()

        fit_nn_potentials(model, x_train, y_train, num_epochs=num_epochs, use_cuda=use_cuda)

    if use_cuda:
        model.cuda()
    model.eval()

    num_examples = args.print_examples
    if num_examples > 0:
        names_examples = names_test[:num_examples]
        x_examples = x_test[:num_examples]
        y_examples = y_test[:num_examples]

    A = np.zeros((8,3), dtype=np.float32)
    I = np.zeros(8, dtype=int)
    # helix
    A[0,0] = 1.0
    A[3,0] = 1.0
    A[4,0] = 1.0
    I[0] = 0
    I[3] = 0
    I[4] = 0
    # sheet
    A[1,1] = 1.0
    A[2,1] = 1.0
    I[1] = 1
    I[2] = 1
    # coil
    A[5,2] = 1.0
    A[6,2] = 1.0
    A[7,2] = 1.0
    I[5] = 2
    I[6] = 2
    I[7] = 2

    A = torch.from_numpy(A)
    I = torch.from_numpy(I)
    if use_cuda:
        A = A.cuda()
        I = I.cuda()

    n = 0
    acc_8 = 0
    acc_3 = 0
    loss_8 = 0
    loss_3 = 0

    x_test = torch.cat([torch.from_numpy(x) for x in x_test], 0)
    y_test = torch.cat([torch.from_numpy(y).long() for y in y_test], 0)

    if use_cuda and not args.all_hidden:
        x_test = x_test.cuda()
        y_test = y_test.cuda()

    mb = 256
    with torch.no_grad():
        for i in range(0, len(x_test), mb):
            x = x_test[i:i+mb]
            y = y_test[i:i+mb]

            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            potentials = model(x).view(x.size(0), -1)

            ## 8-class SS
            l = F.cross_entropy(potentials, y).item()
            _,y_hat = potentials.max(1)
            correct = torch.sum((y == y_hat).float()).item()

            n += x.size(0)
            delta = x.size(0)*(l - loss_8)
            loss_8 += delta/n
            delta = correct - x.size(0)*acc_8
            acc_8 += delta/n

            ## 3-class SS
            y = I[y]
            p = F.softmax(potentials, 1) 
            p = torch.mm(p, A) # ss3 probabilities
            log_p = torch.log(p)
            l = F.nll_loss(log_p, y).item()
            _,y_hat = log_p.max(1)
            correct = torch.sum((y == y_hat).float()).item()

            delta = x.size(0)*(l - loss_3)
            loss_3 += delta/n
            delta = correct - x.size(0)*acc_3
            acc_3 += delta/n
                
    print('-', '-', '8-class', '-', '3-class', '-')
    print('split', 'perplexity', 'accuracy', 'perplexity', 'accuracy')
    print('test', np.exp(loss_8), acc_8, np.exp(loss_3), acc_3)


    if num_examples > 0:
        for i in range(num_examples):
            name = names_examples[i].decode('utf-8')
            x = x_examples[i]
            y = y_examples[i]

            seq = sequences_test[i]

            print('>' + name + ' sequence')
            print(seq)
            print('')

            ss = ''.join(secstr[c] for c in y)
            ss = ss.replace(' ', 'C')
            print('>' + name + ' secstr')
            print(ss)
            print('')

            x = torch.from_numpy(x)
            if use_cuda:
                x = x.cuda()
            potentials = model(x)
            _,y_hat = torch.max(potentials, 1)
            y_hat = y_hat.cpu().numpy()

            ss_hat = ''.join(secstr[c] for c in y_hat)
            ss_hat = ss_hat.replace(' ', 'C')
            print('>' + name + ' predicted')
            print(ss_hat)
            print('')



if __name__ == '__main__':
    main()
