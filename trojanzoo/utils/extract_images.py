# extract marks and/or masks from .npy or .npz files and save them as image files.
import glob
import os
from pathlib import Path

import numpy as np
from numpy import array as npa
from PIL import Image


def load_mask_by_names(root, defense, attack, dataset, ntrig, trial):
    sep = os.path.sep
    pat = rf"{root}\{ntrig}\multitest_results\defenses\{defense}-{attack}-{dataset}"
    folder = pat + '-' + str(trial)
    npy_list = glob.glob(folder + sep + '*best.npy')
    npz_list = glob.glob(folder + sep + '*.npz')
    path = (npy_list + npz_list)[0]
    masks = load_masked_mark(path)
    adj_masks = [im_adjust(mask) for mask in masks]
    adj_masks = [m.max() - m for m in adj_masks]
    return masks, adj_masks


def load_masked_mark_npz(path):
    arr = np.load(path, allow_pickle=True)
    index = 0
    res = []
    while f'mark_list{index}' in arr.files:
        mark = arr[f'mark_list{index}']
        if len(mark.shape) == 4 and mark.shape[0] == 1:
            mark = mark[0, ...]
        if len(mark.shape) == 3:
            mark = mark.transpose(1, 2, 0)
        if f'mask_list{index}' in arr.files:
            mask = arr[f'mask_list{index}']
            mask = np.stack((mask,) * 3, axis=-1)
            final_mark = mark * mask
        else:
            final_mark = mark
        if len(final_mark.shape) == 3 and final_mark.shape[2] == 1:
            final_mark = final_mark[..., 0]
        res.append(final_mark)
        index += 1
    return res


def load_masked_mark_npy(path):
    arr = np.load(path, allow_pickle=True).item()
    index = 0
    res = []
    while index in arr:
        mark = arr[index]['mark']
        if len(mark.shape) == 4 and mark.shape[0] == 1:
            mark = mark[0, ...]
        if len(mark.shape) == 3:
            mark = mark.transpose(1, 2, 0)
        if f'mask' in arr[index]:
            mask = arr[index]['mask']
            mask = np.stack((mask,) * 3, axis=-1)
            final_mark = mark * mask
        else:
            final_mark = mark
        if len(final_mark.shape) == 3 and final_mark.shape[2] == 1:
            final_mark = final_mark[..., 0]
        res.append(final_mark)
        index += 1
    return res


def load_masked_mark(path):
    if path.endswith('.npy'):
        return load_masked_mark_npy(path)
    elif path.endswith('.npz'):
        return load_masked_mark_npz(path)
    else:
        raise ValueError('file should be npz or npy')


def im_adjust(im):
    if len(im.shape) == 3 and im.shape[2] == 3:
        outim = np.zeros_like(im)
        for i in range(3):
            outim[..., i] = channel_adjust(im[..., i])
    else:
        outim = channel_adjust(im)
    return outim


def channel_adjust(im):
    p0 = np.percentile(im, 1)
    p1 = np.percentile(im, 99)
    if p1 - p0 > 1e-6:
        outim = np.interp(im, (p0, p1), (0, 1))
    else:
        outim = im.copy()
    return outim


def save_masks(paths):
    for i, path in enumerate(paths):
        p = Path(path)
        masks = load_masked_mark(path)
        for j, mask in enumerate(masks):
            no_adj_mask = mask.copy()
            no_adj_mask = npa(255 * no_adj_mask, np.uint8)
            no_adj_mask = Image.fromarray(no_adj_mask)
            out_name = p.stem + '_' + str(j) + '_' + 'no_adjust_' + p.suffix[1:] + '.png'
            out_path = os.path.join(p.parent, out_name)
            no_adj_mask.save(out_path)

            mask = im_adjust(mask)
            mask = npa(255 * mask, np.uint8)
            mask = Image.fromarray(mask)
            out_name = p.stem + '_' + str(j) + '_' + p.suffix[1:] + '.png'
            out_path = os.path.join(p.parent, out_name)
            mask.save(out_path)
