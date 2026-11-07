"""DISC ImageNet-subset loader for the ICLR 2027 rebuttal comparison (LinBa on DISC's benchmark).

The tensors are produced by DISC's own preprocessing (DISC repo: rebuttal/export_imagenet_tensors.py ->
DM.utils.get_dataset('ImageNet', ...): ToTensor, ImageNet mean/std normalisation, Resize(res), CenterCrop(res);
labels mapped to 0..9 with DISC's class_map), so LinBa trains and is tested on exactly the images DISC uses.

File: $LINBA_DISC_DATA/<dataset>.pt with keys x_train (N,3,H,W float), y_train (N,), x_test, y_test.
"""
import os
import torch
from torch.utils.data import TensorDataset


def load_disc_imagenet(dataset, validation_ratio=0):
    root = os.environ.get('LINBA_DISC_DATA')
    assert root, 'set LINBA_DISC_DATA to the directory holding <dataset>.pt'
    d = torch.load(os.path.join(root, dataset + '.pt'), map_location='cpu')
    x_tr, y_tr = d['x_train'].float(), d['y_train'].long()
    x_te, y_te = d['x_test'].float(), d['y_test'].long()
    if validation_ratio > 0:
        # held-out split of the training set (the official README tunes with validation_ratio=0.1)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(y_tr), generator=g)
        n_val = int(len(y_tr) * validation_ratio)
        x_te, y_te = x_tr[perm[:n_val]], y_tr[perm[:n_val]]
        x_tr, y_tr = x_tr[perm[n_val:]], y_tr[perm[n_val:]]
    channel, im_size = x_tr.shape[1], (x_tr.shape[2], x_tr.shape[3])
    print(f'[disc_imagenet] {dataset}: train {tuple(x_tr.shape)} test {tuple(x_te.shape)} val_ratio={validation_ratio}')
    return channel, im_size, 10, 10, TensorDataset(x_tr, y_tr), TensorDataset(x_te, y_te)
