from __future__ import print_function,division

import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.utils.data

from bepler.alphabets import Uniprot21
from bepler.parse_utils import parse_3line
import bepler.transmembrane as tm

def load_3line(path, alphabet):
    with open(path, 'rb') as f:
        names, x, y = parse_3line(f)
    x = [alphabet.encode(x) for x in x]
    y = [tm.encode_labels(y) for y in y]
    return x, y

def load_data():
    alphabet = Uniprot21()

    path = 'data/transmembrane/TOPCONS2_datasets/TM.3line'
    x_tm, y_tm = load_3line(path, alphabet)

    path = 'data/transmembrane/TOPCONS2_datasets/SP+TM.3line'
    x_tm_sp, y_tm_sp = load_3line(path, alphabet)

    path = 'data/transmembrane/TOPCONS2_datasets/Globular.3line'
    x_glob, y_glob = load_3line(path, alphabet)

    path = 'data/transmembrane/TOPCONS2_datasets/Globular+SP.3line'
    x_glob_sp, y_glob_sp = load_3line(path, alphabet)

    datasets = {'TM': (x_tm, y_tm), 'SP+TM': (x_tm_sp, y_tm_sp),
                'Globular': (x_glob, y_glob), 'Globular+SP': (x_glob_sp, y_glob_sp)}

    return datasets

def split_dataset(xs, ys, random=np.random, k=5):
    x_splits = [[] for _ in range(k)]
    y_splits = [[] for _ in range(k)]
    order = random.permutation(len(xs))
    for i in range(len(order)):
        j = order[i]
        x_s = x_splits[i%k]
        y_s = y_splits[i%k]
        x_s.append(xs[j])
        y_s.append(ys[j])
    return x_splits, y_splits

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
            #setattr(layer, dest, getattr(lstm, bepler))
            
            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
            #setattr(layer, dest, getattr(lstm, bepler))
        layers.append(layer)
        in_size = 2*hidden_dim
    return layers

def featurize(x, lm_embed, lstm_stack, proj, include_lm=True, lm_only=False):
    zs = []
    
    x_onehot = x.new(x.size(0),x.size(1), 21).float().zero_()
    x_onehot.scatter_(2,x.unsqueeze(2),1)
    zs.append(x_onehot)
    
    h = lm_embed(x)
    if include_lm:
        zs.append(h)
    if not lm_only:
        for lstm in lstm_stack:
            h,_ = lstm(h)
            zs.append(h)
        h = proj(h.squeeze(0)).unsqueeze(0)
        zs.append(h)
    z = torch.cat(zs, 2)
    return z

def featurize_dict(datasets, lm_embed, lstm_stack, proj, use_cuda=False, include_lm=True, lm_only=False):
    z = {}
    for k,v in datasets.items():
        x_k = v[0]
        z[k] = []
        with torch.no_grad():
            for x in x_k:
                x = torch.from_numpy(x).long().unsqueeze(0)
                if use_cuda:
                    x = x.cuda()
                z_x = featurize(x, lm_embed, lstm_stack, proj, include_lm=include_lm, lm_only=lm_only)
                z_x = z_x.squeeze(0).cpu()
                z[k].append(z_x)
    return z

def featurize_one_hot_dict(datasets, n):
    z = {}
    for k,v in datasets.items():
        x_k = v[0]
        z[k] = []
        with torch.no_grad():
            for x in x_k:
                x = torch.from_numpy(x).long()
                one_hot = torch.FloatTensor(x.size(0), n).to(x.device)
                one_hot.zero_()
                one_hot.scatter_(1, x.unsqueeze(1), 1)
                z[k].append(one_hot)
    return z

def make_train_test(splits, j, k):
    x_train = []
    y_train = []
    for v in splits.values():
        for i in range(k):
            if i != j:
                x_train += v[0][i]
                y_train += v[1][i]

    x_test = {k:v[0][j] for k,v in splits.items()}
    y_test = {k:v[1][j] for k,v in splits.items()}

    return x_train, y_train, x_test, y_test

class ListDataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

class LSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*n_hidden, n_out)
        
    def forward(self, x):
        if type(x) is not PackedSequence:
            ndim = len(x.size())
            if ndim == 2:
                x = x.unsqueeze(0)
        h,_ = self.rnn(x)
        if type(h) is PackedSequence:
            z = self.linear(h.data)
            return PackedSequence(z, h.batch_sizes)
        else:
            z = self.linear(h.view(h.size(0)*h.size(1), -1))
            z = z.view(h.size(0), h.size(1), -1)
            if ndim == 2:
                z = z.squeeze(0)
            return z

def train(x_train, y_train, num_epochs=10, hidden_dim=100, use_cuda=False):

    d = x_train[0].size(1)

    model = LSTM(d, hidden_dim, 4)
    if use_cuda:
        model.cuda()

    batch_size = 1
    dataset = ListDataset(x_train, y_train)
    iterator = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

    optim = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        for x,y in iterator:
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            log_p = model(x).squeeze(0)
            x = x.squeeze(0)
            y = y.squeeze(0)
            loss = F.cross_entropy(log_p, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

    return model

def evaluate_model(model, grammar, z_test, y_test, use_cuda=False):
    results = {}
    for key in z_test:
        y_hats = grammar.predict_viterbi(z_test[key], model, use_cuda)
        correct = np.zeros(len(y_hats))
        for i,(pred,target) in enumerate(zip(y_hats, y_test[key])):
            correct[i] = tm.is_prediction_correct(pred, target)
        results[key] = correct.mean()
    overall = sum(results.values())/len(results)
    return overall, results

def evaluate_split(splits, j, k, num_epochs=10, hidden_dim=100, use_cuda=False, grammar=tm.Grammar()):
    x_train, y_train, x_test, y_test = make_train_test(splits, j, k)
    model = train(x_train, y_train, num_epochs=num_epochs, hidden_dim=hidden_dim, use_cuda=use_cuda)
    model.eval()
    overall,results = evaluate_model(model, grammar, x_test, y_test, use_cuda=use_cuda)
    return overall, results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to saved embedding model')
    parser.add_argument('--hidden-dim', type=int, default=150, help='dimension of LSTM (default: 150)')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of training epochs (default: 10)')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    args = parser.parse_args()

    datasets = load_data()
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim

    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)


    ## load the embedding model
    if args.model == '1-hot':
        print('# featurizing data', file=sys.stderr)
        z = featurize_one_hot_dict(datasets, 21)
        datasets = {k: (z[k],v[1]) for k,v in datasets.items()}

    else:
        encoder = torch.load(args.model)
        encoder.eval()
        encoder = encoder.embedding

        lm_embed = encoder.embed
        lstm_stack = unstack_lstm(encoder.rnn)
        proj = encoder.proj

        if use_cuda:
            lm_embed.cuda()
            for lstm in lstm_stack:
                lstm.cuda()
            proj.cuda()

        ## featurize the sequences
        print('# featurizing data', file=sys.stderr)
        z = featurize_dict(datasets, lm_embed, lstm_stack, proj, use_cuda=use_cuda)

        del lm_embed
        del lstm_stack
        del proj
        del encoder

        datasets = {k: (z[k],v[1]) for k,v in datasets.items()}

    ## split into folds
    random = np.random.RandomState(10)
    K = 10
    datasets_split = {k: split_dataset(v[0], v[1], random=random, k=K) for k,v in datasets.items()}

    ## train/test on each fold
    print('# training and evaluating with', K, 'folds', file=sys.stderr)
    print('# using', hidden_dim, 'LSTM units', file=sys.stderr)
    tags = ['TM', 'SP+TM', 'Globular', 'Globular+SP']
    print('\t'.join(['Fold'] + tags + ['Overall']))
    split_results = {}
    split_overall = []
    for i in range(K):
        overall, results = evaluate_split(datasets_split, i, K,
                                          num_epochs=num_epochs, hidden_dim=hidden_dim,
                                          use_cuda=use_cuda)
        for key in tags:
            this = split_results.get(key, [])
            this.append(results[key])
            split_results[key] = this
        split_overall.append(overall)
        cols = [str(i)] + ['{:.5f}'.format(results[key]) for key in tags] + ['{:.5f}'.format(overall)]
        line = '\t'.join(cols)
        print(line)

    results = {key:np.mean(values) for key,values in split_results.items()}
    overall = np.mean(split_overall)

    cols = ['All'] + ['{:.5f}'.format(results[key]) for key in tags] + ['{:.5f}'.format(overall)]
    line = '\t'.join(cols)
    print(line)


if __name__ == '__main__':
    main()



