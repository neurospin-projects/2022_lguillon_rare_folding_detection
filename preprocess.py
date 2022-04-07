# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

import os

import numpy as np
import pandas as pd
import random
import torch
import torchvision.transforms as transforms
from scipy.ndimage import rotate, find_objects, shift
from scipy.special import expit
from soma import aims

import deep_folding as df
from config import Config


class SkeletonDataset():
    """Custom dataset for skeleton images that includes image file paths.

    Args:
        dataframe: dataframe containing training and testing arrays
        filenames: optional, list of corresponding filenames

    Returns:
        tuple_with_path: tuple of type (sample, filename) with sample normalized
                         and padded
    """
    def __init__(self, dataframe, filenames=None, min_size=100,
                 visu_check=False):
        torch.manual_seed(17)
        self.df = dataframe
        self.filenames = filenames
        self.visu_check = visu_check
        self.min_size = min_size
        self.config = Config()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.config.model == 'vae':
            if self.filenames:
                #filename = self.filenames[idx]
                sample = np.expand_dims(np.squeeze(self.df.iloc[idx]['distmaps']), axis=0)
                filename = self.df.iloc[idx]['subjects']

            fill_value = 0
            """self.transform = transforms.Compose([RotateTensor(filename), NormalizeSkeleton(),
                                transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
                                Padding(list(self.config.in_shape), fill_value=fill_value)
                                   ])"""
            self.transform = transforms.Compose([RotateTensor(filename), NormalizeSkeleton(),
                                Padding(list(self.config.in_shape), fill_value=fill_value)
                                   ])
            """self.transform = transforms.Compose([NormalizeSkeleton(),
                                Padding(list(self.config.in_shape), fill_value=fill_value)
                                   ])"""
            """self.transform = transforms.Compose([Padding(list(self.config.in_shape),
                                                 fill_value=fill_value)
                                                ])"""
            tuple_with_path = (self.transform(sample), filename)
            #print(np.unique(tuple_with_path[0]))
            #np.save(f"/neurospin/dico/lguillon/miccai_22/aug_{filename}.npy", tuple_with_path[0])
            return tuple_with_path

        else:
            if self.filenames:
                #filename = self.filenames[idx]
                sample = np.expand_dims(np.squeeze(self.df.iloc[idx]['skeleton']), axis=0)
                filename = self.df.iloc[idx]['subjects']
                labels = np.expand_dims(np.squeeze(self.df.iloc[idx]['labels']), axis=0)

            fill_value = 0
            self.transform = transforms.Compose([randomSuppression(labels, self.min_size),
                             NormalizeSkeleton(),
                             Padding(list(self.config.in_shape), fill_value=fill_value)
                             ])
            transf_sample, target = self.transform(np.copy(sample))

            padding = transforms.Compose([
                     Padding(list(self.config.in_shape), fill_value=fill_value)])

            tuple_with_path = (transf_sample, filename)

            if self.visu_check:
                self.transform_1 = transforms.Compose([NormalizeSkeleton(),
                                   Padding(list(self.config.in_shape), fill_value=fill_value)
                                   ])
                orig_sample = self.transform_1(sample)
                return tuple_with_path, padding(target), orig_sample

            else:
                return tuple_with_path, padding(target)


class RotateTensor(object):
    """Apply a random rotation on the images
    """

    def __init__(self, filename, max_angle=45):
        torch.manual_seed(17)
        self.config = Config()
        self.filename = filename

        self.max_angle = max_angle

        # Load intermediate bigger crop
        sub_id = np.load(os.path.join(self.config.aug_dir, 'Rdistmaps/sub_id.npy'))
        orig_file = np.load(os.path.join(self.config.aug_dir, "Rdistmaps/distmap_1mm.npy"), \
                            mmap_mode='r')
        idx = np.where(sub_id==filename)
        self.orig_img = orig_file[idx]
        #print(self.orig_img.shape)
        vol_test = aims.Volume(self.orig_img)
        aims.write(vol_test, f"/neurospin/dico/lguillon/distmap/rot_test_{self.filename}.nii.gz")

        # Load cropped mask
        mask = aims.read(os.path.join(self.config.aug_dir,
                                      'mask_distmap_cropped.nii.gz'))
        self.mask = np.asarray(mask)
        self.mask = np.squeeze(self.mask)

    def __call__(self, arr):
        #print('rotation')
        #arr = self.orig_img[0,:, :, :]
        arr = np.squeeze(self.orig_img)
        arr_shape = arr.shape
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        #angle = 45
        arr_rot = rotate(arr, angle, reshape=False)

        #print(1, arr_rot.shape)

        arr_rot[self.mask==0] = 0
        #print(2, arr_rot.shape)
        arr_rot = np.expand_dims(arr_rot, axis=0)
        #print(3, arr_rot.shape)

        #vol = aims.Volume(np.expand_dims(np.squeeze(arr_rot), axis=3))
        #arr_rot = aims.VolumeView(vol, self.bbmin, self.bbmax-self.bbmin)
        #print(self.bbmin, self.bbmax-self.bbmin)
        #np.save(f"/neurospin/dico/lguillon/distmap/rot_test_{self.filename}.npy", arr_rot)
        #aims.write(vol, f"/neurospin/dico/lguillon/distmap/rot_test_{self.filename}.nii.gz")
        #print(3, np.asarray(arr_rot).shape)

        #return torch.from_numpy(np.asarray(arr_rot))
        return torch.from_numpy(arr_rot)


class RotateShiftTensor(object):
    """Apply a random rotation on the images
    """

    def __init__(self, filename, max_angle=45):
        torch.manual_seed(17)
        self.config = Config()
        self.filename = filename

        self.max_angle = max_angle

        # Load intermediate bigger crop
        sub_id = np.load(os.path.join(self.config.aug_dir, 'Rdistmaps/sub_id.npy'))
        orig_file = np.load(os.path.join(self.config.aug_dir, "Rdistmaps/distmap_1mm.npy"), \
                            mmap_mode='r')
        idx = np.where(sub_id==filename)
        self.orig_img = orig_file[idx]
        #print(self.orig_img.shape)
        vol_test = aims.Volume(self.orig_img)
        aims.write(vol_test, f"/neurospin/dico/lguillon/distmap/rot_test_{self.filename}.nii.gz")

        # Load cropped mask
        mask = aims.read(os.path.join(self.config.aug_dir,
                                      'mask_distmap_cropped.nii.gz'))
        self.mask = np.asarray(mask)
        self.mask = np.squeeze(self.mask)

    def __call__(self, arr):
        #print('rotation')
        #arr = self.orig_img[0,:, :, :]
        arr = np.squeeze(self.orig_img)
        arr_shape = arr.shape
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        #angle = 45
        arr_rot = rotate(arr, angle, reshape=False)
        arr_rot = shift(arr_rot, 5)

        #print(1, arr_rot.shape)

        arr_rot[self.mask==0] = 0
        #print(2, arr_rot.shape)
        arr_rot = np.expand_dims(arr_rot, axis=0)
        #print(3, arr_rot.shape)

        #vol = aims.Volume(np.expand_dims(np.squeeze(arr_rot), axis=3))
        #arr_rot = aims.VolumeView(vol, self.bbmin, self.bbmax-self.bbmin)
        #print(self.bbmin, self.bbmax-self.bbmin)
        #np.save(f"/neurospin/dico/lguillon/distmap/rot_test_{self.filename}.npy", arr_rot)
        #aims.write(vol, f"/neurospin/dico/lguillon/distmap/rot_test_{self.filename}.nii.gz")
        #print(3, np.asarray(arr_rot).shape)

        #return torch.from_numpy(np.asarray(arr_rot))
        return torch.from_numpy(arr_rot)


class randomSuppression(object):
    """
    """
    def __init__(self, foldlabel_map, min_size=100):
        #self.sample = sample
        self.foldlabel_map = foldlabel_map
        self.min_size = min_size

    def random_choice(self):
        self.del_list = []
        total_del = 0
        folds_list = np.unique(self.foldlabel_map, return_counts=True)
        folds_dico = {key: value for key, value in zip(folds_list[0], folds_list[1])}

        # We don't take into account the background in the random choice of fold
        folds_dico.pop(0, None)

        # Random choice of fold
        fold = random.choice(list(folds_dico.keys()))
        #print(folds_dico[fold])

        # if fold size < 100, deletion of other folds
        if folds_dico[fold]<self.min_size:
            if folds_dico[fold]>5 :
                total_del = folds_dico[fold]
                self.del_list.append(fold)
                folds_dico.pop(fold, None)
            while total_del <= self.min_size:
                fold = random.choice([k for k, v in folds_dico.items() if v>5])
                total_del += folds_dico[fold]
                self.del_list.append(fold)
                folds_dico.pop(fold, None)

        # if fold size >= 100, suppression of this fold only
        else:
            self.del_list.append(fold)

    def __call__(self, sample):
        target = np.zeros(list(sample.shape))
        assert(target.shape==sample.shape)

        self.random_choice()

        for fold in self.del_list:
            self.foldlabel_map[self.foldlabel_map==fold] = 9999

        ## suppression of chosen folds
        old = np.unique(sample, return_counts=True)
        sample[self.foldlabel_map==9999] = 0
        new = np.unique(sample, return_counts=True)
        assert(new[1][0]-old[1][0]>=self.min_size-50)

        ## writing of deleted folds to reconstruct in target
        target[self.foldlabel_map==9999] = 1

        return sample, target


class NormalizeSkeleton(object):
    """
    Class to normalize skeleton objects,
    black voxels: 0
    grey and white voxels: 1
    """
    def __init__(self, distmap=True, nb_cls=None):
        """ Initialize the instance"""
        self.nb_cls = nb_cls
        self.distmap = distmap

    def __call__(self, arr):
        if self.nb_cls==2:
            arr[arr > 0] = 1
        elif self.distmap==True:
            arr[arr!=0] = expit(arr[arr!=0])
            # arr[arr!=0] = np.tanh(arr[arr!=0])

        return arr


class Padding(object):
    """ Apply a padding.
    Parameters
    ----------
    arr: array
        the input data.
    shape: list of int
        the desired shape.
    fill_value: int, default 0
        the value used to fill the array.
    Returns
    -------
    transformed: array
        the transformed input data.
    """
    def __init__(self, shape, fill_value=0):
        self.shape = shape
        self.fill_value = fill_value

    def __call__(self, arr):
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append((half_shape_i, half_shape_i))
            else:
                padding.append((half_shape_i, half_shape_i + 1))
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append((0, 0))
        return np.pad(arr, padding, mode="constant",
                      constant_values=self.fill_value)
