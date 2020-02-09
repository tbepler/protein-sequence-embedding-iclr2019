from __future__ import print_function,division

import numpy as np
import pandas as pd
import sys
import os
import glob
from PIL import Image
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
from src.utils import ContactMapDataset, collate_lists
from src.utils import PairedDataset, AllPairsDataset, collate_paired_sequences
from src.utils import MultinomialResample
import src.models.embedding
import src.models.multitask
from src.metrics import average_precision

cmap_paths = glob.glob('data/SCOPe/pdbstyle-2.06/*/*.png')
cmap_dict = {os.path.basename(path)[:7] : path for path in cmap_paths}


def load_data(path, alphabet):
    with open(path, 'rb') as f:
        names, structs, sequences = scop.parse_astral(f, encoder=alphabet)    
    x = [torch.from_numpy(x).long() for x in sequences]
    s = torch.from_numpy(structs)
    c = []
    for name in names:
        name = name.decode('utf-8')
        if name not in cmap_dict:
            name = 'd' + name[1:]
        path = cmap_dict[name]
        im = np.array(Image.open(path), copy=False)
        contacts = np.zeros(im.shape, dtype=np.float32)
        contacts[im == 1] = -1
        contacts[im == 255] = 1
        # mask the matrix below the diagonal
        mask = np.tril_indices(contacts.shape[0], k=-1)
        contacts[mask] = -1
        c.append(torch.from_numpy(contacts))
    return x, s, c 


def load_scop_testpairs(astral_testpairs_path, alphabet):
    print('# loading test sequence pairs:', astral_testpairs_path, file=sys.stderr)
    test_pairs_table = pd.read_csv(astral_testpairs_path, sep='\t') 
    x0_test = [x.encode('utf-8').upper() for x in test_pairs_table['sequence_A']]
    x0_test = [torch.from_numpy(alphabet.encode(x)).long() for x in x0_test]
    x1_test = [x.encode('utf-8').upper() for x in test_pairs_table['sequence_B']]
    x1_test = [torch.from_numpy(alphabet.encode(x)).long() for x in x1_test]
    y_test = test_pairs_table['similarity'].values
    y_test = torch.from_numpy(y_test).long()

    return x0_test, x1_test, y_test


def similarity_grad(model, x0, x1, y, use_cuda, weight=0.5):
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

    # backprop weighted loss
    w_loss = loss*weight
    w_loss.backward()

    # calculate minibatch performance metrics
    with torch.no_grad():
        p = F.sigmoid(logits) 
        ones = p.new(b,1).zero_() + 1
        p_ge = torch.cat([ones, p], 1)
        p_lt = torch.cat([1-p, ones], 1)
        p = p_ge*p_lt
        p = p/p.sum(1,keepdim=True) # make sure p is normalized

        _,y_hard = torch.max(p, 1)
        levels = torch.arange(5).to(p.device)
        y_hat = torch.sum(p*levels.float(), 1)
        y = torch.sum(y.data, 1)

        loss = F.cross_entropy(p, y).item() # calculate cross entropy loss from p vector

        correct = torch.sum((y == y_hard).float()).item()
        mse = torch.mean((y.float() - y_hat)**2).item()

    return loss, correct, mse, b


def contacts_grad(model, x, y, use_cuda, weight=0.5):
    b = len(x)
    x,order = pack_sequences(x)
    x = PackedSequence(Variable(x.data), x.batch_sizes)
    z = model(x) # embed the sequences
    z = unpack_sequences(z, order)

    logits = []
    for i in range(b):
        zi = z[i]
        lp = model.predict(zi.unsqueeze(0)).view(-1)
        logits.append(lp)
    logits = torch.cat(logits, 0)

    y = torch.cat([yi.view(-1) for yi in y])
    if use_cuda:
        y = y.cuda()
    mask = (y < 0)

    logits = logits[~mask]
    y = Variable(y[~mask])
    b = y.size(0)

    loss = F.binary_cross_entropy_with_logits(logits, y)

    # backprop weighted loss
    w_loss = loss*weight
    w_loss.backward()

    # calculate the recall and precision
    with torch.no_grad():
        p_hat = F.sigmoid(logits)
        tp = torch.sum(p_hat*y).item()
        gp = y.sum().item()
        pp = p_hat.sum().item()

    return loss.item(), tp, gp, pp, b


def predict_contacts(model, x, y, use_cuda):
    b = len(x)
    x,order = pack_sequences(x)
    x = PackedSequence(Variable(x.data), x.batch_sizes)
    z = model(x) # embed the sequences
    z = unpack_sequences(z, order)

    logits = []
    y_list = []
    for i in range(b):
        zi = z[i]
        lp = model.predict(zi.unsqueeze(0)).view(-1)

        yi = y[i].view(-1)
        if use_cuda:
            yi = yi.cuda()
        mask = (yi < 0)

        lp = lp[~mask]
        yi = yi[~mask]

        logits.append(lp)
        y_list.append(yi)

    return logits, y_list


def eval_contacts(model, test_iterator, use_cuda):
    logits = []
    y = []

    for x,y_mb in test_iterator:
        logits_this, y_this = predict_contacts(model, x, y_mb, use_cuda)
        logits += logits_this
        y += y_this

    y = torch.cat(y, 0)
    logits = torch.cat(logits, 0)

    loss = F.binary_cross_entropy_with_logits(logits, y).item()

    p_hat = F.sigmoid(logits)
    tp = torch.sum(y*p_hat).item()
    pr = tp/torch.sum(p_hat).item()
    re = tp/torch.sum(y).item()
    f1 = 2*pr*re/(pr + re)            

    y = y.cpu().numpy()
    logits = logits.data.cpu().numpy()

    aupr = average_precision(y, logits)

    return loss, pr, re, f1, aupr

def eval_similarity(model, test_iterator, use_cuda):
    y = []
    logits = []
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
    y_hat = torch.sum(p*levels.float(), 1)

    accuracy = torch.mean((y == y_hard).float()).item()
    mse = torch.mean((y.float() - y_hat)**2).item()

    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    r,_ = pearsonr(y_hat, y)
    rho,_ = spearmanr(y_hat, y)

    return loss, accuracy, mse, r, rho


def main():
    import argparse
    parser = argparse.ArgumentParser('Script for training contact prediction model')

    parser.add_argument('--dev', action='store_true', help='use train/dev split')

    parser.add_argument('--rnn-type', choices=['lstm', 'gru'], default='lstm', help='type of RNN block to use (default: lstm)')
    parser.add_argument('--embedding-dim', type=int, default=100, help='embedding dimension (default: 40)')
    parser.add_argument('--input-dim', type=int, default=512, help='dimension of input to RNN (default: 512)')
    parser.add_argument('--rnn-dim', type=int, default=512, help='hidden units of RNNs (default: 128)')
    parser.add_argument('--num-layers', type=int, default=3, help='number of RNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability (default: 0)')


    parser.add_argument('--hidden-dim', type=int, default=50, help='number of hidden units for comparison layer in contact predictionn (default: 50)')
    parser.add_argument('--width', type=int, default=7, help='width of convolutional filter for contact prediction (default: 7)')


    parser.add_argument('--epoch-size', type=int, default=100000, help='number of examples per epoch (default: 100,000)')
    parser.add_argument('--epoch-scale', type=int, default=5, help='report heldout performance every this many epochs (default: 5)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs (default: 100)')


    parser.add_argument('--similarity-batch-size', type=int, default=64, help='minibatch size for similarity prediction loss in pairs (default: 64)')
    parser.add_argument('--contact-batch-size', type=int, default=10, help='minibatch size for contact predictionn loss (default: 10)')


    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.5, help='weight on the similarity objective, contact map objective weight is one minus this (default: 0.5)')

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
    alphabet = Uniprot21()

    astral_train_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.fa'
    astral_test_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.fa'
    astral_testpairs_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.sampledpairs.txt'
    if args.dev:
        astral_train_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.train.fa'
        astral_test_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.dev.fa'
        astral_testpairs_path = 'data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.dev.sampledpairs.txt'


    print('# loading training sequences:', astral_train_path, file=sys.stderr)
    x_train, structs_train, contacts_train = load_data(astral_train_path, alphabet)
    if use_cuda:
        x_train = [x.cuda() for x in x_train]
        #contacts_train = [c.cuda() for c in contacts_train]
    print('# loaded', len(x_train), 'training sequences', file=sys.stderr)

    print('# loading test sequences:', astral_test_path, file=sys.stderr)
    x_test, _, contacts_test = load_data(astral_test_path, alphabet)
    if use_cuda:
        x_test = [x.cuda() for x in x_test]
        #contacts_test = [c.cuda() for c in contacts_test]
    print('# loaded', len(x_test), 'contact map test sequences', file=sys.stderr)

    x0_test, x1_test, y_scop_test = load_scop_testpairs(astral_testpairs_path, alphabet)
    if use_cuda:
        x0_test = [x.cuda() for x in x0_test]
        x1_test = [x.cuda() for x in x1_test]
    print('# loaded', len(x0_test), 'scop test pairs', file=sys.stderr)

    ## make the dataset iterators

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

    # SCOP structural similarity datasets
    scop_levels = torch.cumprod((structs_train.unsqueeze(1) == structs_train.unsqueeze(0)).long(), 2)
    scop_train = AllPairsDataset(x_train, scop_levels, augment=augment)
    scop_test = PairedDataset(x0_test, x1_test, y_scop_test)

    # contact map datasets
    cmap_train = ContactMapDataset(x_train, contacts_train, augment=augment)
    cmap_test = ContactMapDataset(x_test, contacts_test)

    # iterators for contacts data
    batch_size = args.contact_batch_size
    cmap_train_iterator = torch.utils.data.DataLoader(cmap_train
                                                    , batch_size=batch_size
                                                    , shuffle=True
                                                    , collate_fn=collate_lists
                                                    )
    cmap_test_iterator = torch.utils.data.DataLoader(cmap_test
                                                   , batch_size=batch_size
                                                   , collate_fn=collate_lists
                                                   )

    # make the SCOP training iterator have same number of minibatches
    num_steps = len(cmap_train_iterator)
    batch_size = args.similarity_batch_size
    epoch_size = num_steps*batch_size

    similarity = scop_levels.numpy().sum(2)
    levels,counts = np.unique(similarity, return_counts=True)
    order = np.argsort(levels)
    levels = levels[order]
    counts = counts[order]

    tau = args.tau
    print('# using tau:', tau, file=sys.stderr)
    print('#', counts**tau/np.sum(counts**tau), file=sys.stderr)
    weights = counts**tau/counts
    weights = weights[similarity].ravel()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, epoch_size)
    N = epoch_size

    # iterators for similarity data
    scop_train_iterator = torch.utils.data.DataLoader(scop_train
                                                    , batch_size=batch_size
                                                    , sampler=sampler
                                                    , collate_fn=collate_paired_sequences
                                                    )
    scop_test_iterator = torch.utils.data.DataLoader(scop_test
                                                   , batch_size=batch_size
                                                   , collate_fn=collate_paired_sequences
                                                   )

    report_steps = args.epoch_scale


    ## initialize the model 
    rnn_type = args.rnn_type
    rnn_dim = args.rnn_dim
    num_layers = args.num_layers

    embedding_size = args.embedding_dim
    input_dim = args.input_dim
    dropout = args.dropout
    
    print('# initializing embedding model with:', file=sys.stderr)
    print('# embedding_size:', embedding_size, file=sys.stderr)
    print('# input_dim:', input_dim, file=sys.stderr)
    print('# rnn_dim:', rnn_dim, file=sys.stderr)
    print('# num_layers:', num_layers, file=sys.stderr)
    print('# dropout:', dropout, file=sys.stderr)

    lm = None
    if args.lm is not None:
        print('# using pretrained LM:', args.lm, file=sys.stderr)
        lm = torch.load(args.lm)
        lm.eval()
        ## do not update the LM parameters
        for param in lm.parameters():
            param.requires_grad = False

    embedding = src.models.embedding.StackedRNN(len(alphabet), input_dim, rnn_dim
                                               , embedding_size, nlayers=num_layers
                                               , dropout=dropout, lm=lm)

    # similarity prediction parameters
    similarity_kwargs = {}

    # contact map prediction parameters
    hidden_dim = args.hidden_dim
    width = args.width
    cmap_kwargs = {'hidden_dim': hidden_dim, 'width': width}
    
    model = src.models.multitask.SCOPCM(embedding, similarity_kwargs=similarity_kwargs,
                                        cmap_kwargs=cmap_kwargs)
    if use_cuda:
        model.cuda()

    ## setup training parameters and optimizer
    num_epochs = args.num_epochs

    weight_decay = args.weight_decay
    lr = args.lr

    print('# training with Adam: lr={}, weight_decay={}'.format(lr, weight_decay), file=sys.stderr)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    scop_weight = args.lambda_
    cmap_weight = 1 - scop_weight

    print('# weighting tasks with SIMILARITY: {:.3f}, CONTACTS: {:.3f}'.format(scop_weight, cmap_weight), file=sys.stderr)

    ## train the model
    print('# training model', file=sys.stderr)

    save_prefix = args.save_prefix
    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')
    digits = int(np.floor(np.log10(num_epochs))) + 1
    tokens = ['sim_loss', 'sim_mse', 'sim_acc', 'sim_r', 'sim_rho'
             ,'cmap_loss', 'cmap_pr', 'cmap_re', 'cmap_f1', 'cmap_aupr']
    line = '\t'.join(['epoch', 'split'] + tokens)
    print(line, file=output)

    prog_template = '# [{}/{}] training {:.1%} sim_loss={:.5f}, sim_acc={:.5f}, cmap_loss={:.5f}, cmap_f1={:.5f}'

    for epoch in range(num_epochs):
        # train epoch
        model.train()

        scop_n = 0
        scop_loss_accum = 0
        scop_mse_accum = 0
        scop_acc_accum = 0

        cmap_n = 0
        cmap_loss_accum = 0
        cmap_pp = 0
        cmap_pr_accum = 0
        cmap_gp = 0
        cmap_re_accum = 0

        for (cmap_x, cmap_y), (scop_x0, scop_x1, scop_y) in zip(cmap_train_iterator, scop_train_iterator):

            # calculate gradients and metrics for similarity part
            loss, correct, mse, b = similarity_grad(model, scop_x0, scop_x1, scop_y, use_cuda, weight=scop_weight)

            scop_n += b
            delta = b*(loss - scop_loss_accum)
            scop_loss_accum += delta/scop_n
            delta = correct - b*scop_acc_accum
            scop_acc_accum += delta/scop_n
            delta = b*(mse - scop_mse_accum)
            scop_mse_accum += delta/scop_n
            
            report = ((scop_n - b)//100 < scop_n//100)
           
            # calculate the contact map prediction gradients and metrics
            loss, tp, gp_, pp_, b = contacts_grad(model, cmap_x, cmap_y, use_cuda, weight=cmap_weight)

            cmap_gp += gp_
            delta = tp - gp_*cmap_re_accum
            cmap_re_accum += delta/cmap_gp

            cmap_pp += pp_
            delta = tp - pp_*cmap_pr_accum
            cmap_pr_accum += delta/cmap_pp

            cmap_n += b
            delta = b*(loss - cmap_loss_accum)
            cmap_loss_accum += delta/cmap_n

            ## update the parameters
            optim.step()
            optim.zero_grad()
            model.clip()

            if report:
                f1 = 2*cmap_pr_accum*cmap_re_accum/(cmap_pr_accum + cmap_re_accum)
                line = prog_template.format(epoch+1, num_epochs, scop_n/N, scop_loss_accum
                                           , scop_acc_accum, cmap_loss_accum, f1)
                print(line, end='\r', file=sys.stderr)
        print(' '*80, end='\r', file=sys.stderr)
        f1 = 2*cmap_pr_accum*cmap_re_accum/(cmap_pr_accum + cmap_re_accum)
        tokens = [ scop_loss_accum, scop_mse_accum, scop_acc_accum, '-', '-'
                 , cmap_loss_accum, cmap_pr_accum, cmap_re_accum, f1, '-']
        tokens = [x if type(x) is str else '{:.5f}'.format(x) for x in tokens]

        line = '\t'.join([str(epoch+1).zfill(digits), 'train'] + tokens)
        print(line, file=output)
        output.flush()

        # eval and save model
        if (epoch+1) % report_steps == 0:
            model.eval()
            with torch.no_grad():
                scop_loss, scop_acc, scop_mse, scop_r, scop_rho = \
                        eval_similarity(model, scop_test_iterator, use_cuda)
                cmap_loss, cmap_pr, cmap_re, cmap_f1, cmap_aupr = \
                        eval_contacts(model, cmap_test_iterator, use_cuda)

            tokens = [ scop_loss, scop_mse, scop_acc, scop_r, scop_rho
                     , cmap_loss, cmap_pr, cmap_re, cmap_f1, cmap_aupr]
            tokens = ['{:.5f}'.format(x) for x in tokens]

            line = '\t'.join([str(epoch+1).zfill(digits), 'test'] + tokens)
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





