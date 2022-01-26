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

import numpy as np
import pandas as pd
import random
import torch
import torchvision.transforms as transforms

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

        if self.filenames:
            filename = self.filenames[idx]
            sample = np.expand_dims(np.squeeze(self.df.iloc[idx]['skeleton']), axis=0)
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

        # if fold size < 100, deletion of other folds
        if folds_dico[fold]<self.min_size:
            if folds_dico[fold]>5 :
                total_del = folds_dico[fold]
                self.del_list.append(fold)
                folds_dico.pop(fold, None)
            while total_del <= self.min_size:
                #print(total_del)
                fold = random.choice([k for k, v in folds_dico.items() if v<self.min_size and v>5])
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
    def __init__(self, nb_cls=2):
        """ Initialize the instance"""
        self.nb_cls = nb_cls

    def __call__(self, arr):
        if len(arr)>1 and self.nb_cls==2:
            arr[0][arr[0]>0]=1
        else:
            if self.nb_cls==2:
                arr[arr > 0] = 1
        """else:
            arr[arr==40]=30
            arr[arr==70]=80
            arr[arr==30]=1
            arr[arr==60]=2
            arr[arr==80]=3
        print(type(arr))"""
        return arr


class Padding(object):
    """ A class to pad an image.
    """
    def __init__(self, shape, nb_channels=1, fill_value=0):
        """ Initialize the instance.
        Parameters
        ----------
        shape: list of int
            the desired shape.
        nb_channels: int, default 1
            the number of channels.
        fill_value: int or list of int, default 0
            the value used to fill the array, if a list is given, use the
            specified value on each channel.
        """
        self.shape = shape
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels()

    def __call__(self, arr):
        """ Fill an array to fit the desired shape.
        Parameters
        ----------
        arr: np.array
            an input array.
        Returns
        -------
        fill_arr: np.array
            the zero padded array.
        """
        if len(arr)>1:
            if len(arr[0].shape) - len(self.shape) == 1:
                data = []
                for _arr, _fill_value in zip(arr[0], self.fill_value):
                    data.append(self._apply_padding(_arr, _fill_value))
                return np.asarray(data)
            elif len(arr[0].shape) - len(self.shape) == 0:
                return (self._apply_padding(arr[0], self.fill_value), arr[1])
            else:
                raise ValueError("Wrong input shape specified!")
        else:
            if len(arr.shape) - len(self.shape) == 1:
                data = []
                for _arr, _fill_value in zip(arr, self.fill_value):
                    data.append(self._apply_padding(_arr, _fill_value))
                return np.asarray(data)
            elif len(arr.shape) - len(self.shape) == 0:
                return self._apply_padding(arr, self.fill_value)
            else:
                raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, arr, fill_value):
        """ See Padding.__call__().
        """
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

        fill_arr = np.pad(arr, padding, mode="constant",
                          constant_values=fill_value)
        return fill_arr
