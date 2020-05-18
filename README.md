# Learning protein sequence embeddings using information from structure

This repository contains the source code and links to the data and pretrained embedding models accompanying the ICLR 2019 paper: [Learning protein sequence embeddings using information from structure](https://openreview.net/pdf?id=SygLehCqtm)

```
@inproceedings{
bepler2018learning,
title={Learning protein sequence embeddings using information from structure},
author={Tristan Bepler and Bonnie Berger},
booktitle={International Conference on Learning Representations},
year={2019},
}
```

## Setup and dependencies 

Dependencies:
- python 3
- pytorch >= 0.4
- numpy
- scipy
- pandas
- sklearn
- cython
- h5py (for embedding script)

Run setup.py to compile the cython files:

```
python setup.py build_ext --inplace
```

## Data sets

The data sets with train/dev/test splits are provided as .tar.gz files from the links below.

- [SCOPe data](http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/scope.tar.gz)
- [Pfam data](http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/pfam.tar.gz)
- [Protein secondary structure data](http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/secstr.tar.gz)
- [Transmembrane data](http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/transmembrane.tar.gz)
- [CASP12 contact map data](http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/casp12.tar.gz)

The training and evaluation scripts assume that these data sets have been extracted into a directory called 'data'.

## Pretrained models

Our trained versions of the structure-based embedding models and the bidirectional language model can be downloaded [here](http://bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/pretrained_models.tar.gz).

## Author

Tristan Bepler (tbepler@mit.edu)

## Cite

Please cite the above paper if you use this code or pretrained models in your work.

## License

The source code and trained models are provided free for non-commercial use under the terms of the CC BY-NC 4.0 license. See [LICENSE](LICENSE) file and/or https://creativecommons.org/licenses/by-nc/4.0/legalcode for more information.


## Contact

If you have any questions, comments, or would like to report a bug, please file a Github issue or contact me at tbepler@mit.edu.

