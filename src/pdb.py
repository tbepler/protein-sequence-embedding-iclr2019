from __future__ import print_function, division

def parse_secstr_stream(f, comment=b'#'):
    name = None
    flag = -1
    protein = []
    secstr = []
    for line in f:
        if line.startswith(comment):
            continue
        # strip newline 
        line = line.rstrip(b'\r\n')
        if line.startswith(b'>'):
            if name is not None and flag==1:
                yield name, b''.join(protein), b''.join(secstr)
            elif flag == 0:
                assert line[1:].startswith(name)
            
            # each protein has an amino acid sequence
            # and secstr sequence associated with it

            name = line[1:]
            tokens = name.split(b':')
            name = b':'.join(tokens[:-1])
            flag = tokens[-1]

            if flag == b'sequence':
                flag = 0
                protein = []
                secstr = []
            elif flag == b'secstr':
                flag = 1
            else:
                raise Exception("Unrecognized flag: " + flag.decode())

        elif flag==0:
            protein.append(line)
        elif flag==1:
            secstr.append(line)
        else:
            raise Exception("Flag not set properly")

    if name is not None:
        yield name, b''.join(protein), b''.join(secstr)

def parse_secstr(f, comment=b'#'):

    names = []
    proteins = []
    secstrs = []
    for name,protein,secstr in parse_secstr_stream(f, comment=comment):
        names.append(name)
        proteins.append(protein)
        secstrs.append(secstr)
    return names, proteins, secstrs


