from __future__ import print_function,division

import sys
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alphabets import Uniprot21
import src.fasta as fasta
import src.models.sequence


def unstack_lstm(lstm):
    device = next(iter(lstm.parameters())).device

    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        layer.to(device)

        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)
            #setattr(layer, dest, getattr(lstm, src))
            
            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
            #setattr(layer, dest, getattr(lstm, src))
        layer.flatten_parameters()
        layers.append(layer)
        in_size = 2*hidden_dim
    return layers

def embed_stack(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False):
    zs = []
    
    x_onehot = x.new(x.size(0),x.size(1), 21).float().zero_()
    x_onehot.scatter_(2,x.unsqueeze(2),1)
    zs.append(x_onehot)
    
    h = lm_embed(x)
    if include_lm and not final_only:
        zs.append(h)

    if lstm_stack is not None:
        for lstm in lstm_stack:
            h,_ = lstm(h)
            if not final_only:
                zs.append(h)
        h = proj(h.squeeze(0)).unsqueeze(0)
        zs.append(h)

    z = torch.cat(zs, 2)
    return z


def embed_sequence(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False
                  ,  pool='none', use_cuda=False):

    if len(x) == 0:
        return None

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.cuda()

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = embed_stack(x, lm_embed, lstm_stack, proj
                       , include_lm=include_lm, final_only=final_only)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    return z


def load_model(path, use_cuda=False):
    encoder = torch.load(path)
    encoder.eval()

    if use_cuda:
        encoder.cuda()

    if type(encoder) is src.models.sequence.BiLM:
        # model is only the LM
        return encoder.encode, None, None

    encoder = encoder.embedding

    lm_embed = encoder.embed
    lstm_stack = unstack_lstm(encoder.rnn)
    proj = encoder.proj

    return lm_embed, lstm_stack, proj


def main():
    import argparse
    parser = argparse.ArgumentParser('Script for embedding fasta format sequences using a saved embedding model. Saves embeddings as HDF5 file.')

    parser.add_argument('-path', help='sequences to embed in fasta format')
    parser.add_argument('-m', '--model', help='path to saved embedding model')
    parser.add_argument('-o', '--output', help='path to HDF5 output file')
    parser.add_argument('--lm-only', action='store_true', help='only return the language model hidden layers')
    parser.add_argument('--no-lm', action='store_true', help='do not include LM hidden layers in embedding. by default, all hidden layers of all layers are concatenated and returned by this script.')
    parser.add_argument('--proj-only', action='store_true', help='only return the final structure-learned embedding')
    parser.add_argument('--pool', choices=['none', 'sum', 'max', 'avg'], default='none', help='apply some pooling operation over each sequence (default: none)')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')

    args = parser.parse_args()

    path = args.path

    # set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    # load the model
    lm_embed, lstm_stack, proj = load_model(args.model, use_cuda=use_cuda)

    # parse the sequences and embed them
    # write them to hdf5 file
    print('# writing:', args.output, file=sys.stderr)
    h5 = h5py.File(args.output, 'w')

    lm_only = args.lm_only
    if lm_only:
        lstm_stack = None
        proj = None

    no_lm = args.no_lm
    include_lm = not no_lm
    final_only = args.proj_only

    pool = args.pool
    print('# embedding with lm_only={}, no_lm={}, proj_only={}'.format(lm_only, no_lm, final_only), file=sys.stderr)
    print('# pooling:', pool, file=sys.stderr)

    count = 0
    with open(path, 'rb') as f:
        for name,sequence in fasta.parse_stream(f):
            # use sequence name as HDF key
            pid = name.decode('utf-8')
            if len(sequence) == 0:
                print('# WARNING: sequence', pid, 'has length=0. Skipping.', file=sys.stderr)
                continue
            # only do pids we haven't done already...
            if pid not in h5:
                z = embed_sequence(sequence, lm_embed, lstm_stack, proj
                                  , include_lm=include_lm, final_only=final_only
                                  , pool=pool, use_cuda=use_cuda)
                # write as hdf5 dataset
                h5.create_dataset(pid, data=z, compression='lzf')
            count += 1
            print('# {} sequences processed...'.format(count), file=sys.stderr, end='\r')
    print(' '*80, file=sys.stderr, end='\r')
    print('# Done!', file=sys.stderr)



if __name__ == '__main__':
    main()




