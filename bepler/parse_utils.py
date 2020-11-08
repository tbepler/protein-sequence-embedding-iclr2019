from __future__ import print_function,division

def parse_3line(f):
    names = []
    xs = []
    ys = []
    for line in f:
        if line.startswith(b'>'):
            name = line[1:]
            # get the sequence
            x = f.readline().strip()
            # get the transmembrane annotations
            y = f.readline().strip()
            names.append(name)
            xs.append(x)
            ys.append(y)
    return names, xs, ys



