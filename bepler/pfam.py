from __future__ import print_function, division

import numpy as np

def parse_seed(f):
    alignments = []
    a = []
    for line in f:
        if line.startswith(b'#'):
            continue
        if line.startswith(b'//'):
            alignments.append(a)
            a = []
        else:
            _,s = line.split()
            a.append(s)
    if len(a) > 0:
        alignments.append(a)

    return alignments







