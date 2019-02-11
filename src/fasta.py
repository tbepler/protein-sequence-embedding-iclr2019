from __future__ import print_function, division

def parse_stream(f, comment=b'#'):
    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                yield name, b''.join(sequence)
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        yield name, b''.join(sequence)

def parse(f, comment=b'#'):
    names = []
    sequences = []
    name = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                names.append(name)
                sequences.append(b''.join(sequence))
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        names.append(name)
        sequences.append(b''.join(sequence))

    return names, sequences

#def parse(f, comment='#'):
#    names = []
#    sequences = []
#    name = None
#    sequence = []
#    for line in f:
#        if line.startswith(comment):
#            continue
#        line = line.strip()
#        if line.startswith('>'):
#            if name is not None:
#                names.append(name)
#                sequences.append(''.join(sequence))
#            name = line[1:]
#            sequence = []
#        else:
#            sequence.append(line)
#    if name is not None:
#        names.append(name)
#        sequences.append(''.join(sequence))
#
#    return names, sequences


