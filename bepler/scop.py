from __future__ import print_function,division

import numpy as np
import bepler.fasta as fasta

class NullEncoder:
    def encode(self, x):
        return x

def parse_astral_name(name, encode_struct=True):
    tokens = name.split()
    name = tokens[0]

    # encode structure levels as integer by right-padding with zero byte
    # to 4 bytes
    if encode_struct:
        struct = b''
        for s in tokens[1].split(b'.'):
            n = len(s)
            s = s + b'\x00'*(4-n)
            struct += s
        struct = np.frombuffer(struct, dtype=np.int32)
    else:
        struct = np.array(tokens[1].split(b'.'))
    
    return name, struct

def parse_astral(f, encoder=NullEncoder(), encode_struct=True):
    names = []
    structs = []
    sequences = []
    for name,sequence in fasta.parse_stream(f):
        x = encoder.encode(sequence.upper())
        name, struct = parse_astral_name(name, encode_struct=encode_struct)
        names.append(name)
        structs.append(struct)
        sequences.append(x)
    structs = np.stack(structs, 0)
    return names, structs, sequences
