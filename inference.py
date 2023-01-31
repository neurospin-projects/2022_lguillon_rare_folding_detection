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
import shutil

import numpy as np
import pandas as pd
import random
import tempfile
import torch
import torchvision.transforms as transforms
from scipy.ndimage import rotate, find_objects, shift
from scipy.special import expit
from soma import aims, aimsalgo
import dico_toolbox as dtx

import deep_folding as df
from config import Config

from preprocess import convert2Distmap, ApplyMask, Padding, NormalizeSkeleton



class InpaintDatasetTest():
    """Custom dataset for skeleton images that includes image file paths.

    Args:
        dataframe: dataframe containing training and testing arrays
        filenames: optional, list of corresponding filenames

    Returns:
        tuple_with_path: tuple of type (sample, filename) with sample normalized
                         and padded
    """
    def __init__(self, foldlabels, skeletons, distmaps, filenames, data_transforms):
        torch.manual_seed(17)
        self.foldlabel = foldlabels.copy()
        self.skeletons = skeletons.copy()
        self.distmaps = distmaps.copy()
        self.filenames = filenames
        self.data_transforms = data_transforms
        self.config = Config()
        self.flag = True

    def __len__(self):
        return len(self.distmaps)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print(idx)

        if self.config.model == 'vae':
            if len(self.filenames)>0:
                print(1)
                filename = self.filenames[idx]
                print(filename)
                idx_foldlabel = np.where(self.filenames==filename)
                foldlabel = self.foldlabel[idx]
                skeleton = self.skeletons[idx]
                distmap = self.distmaps[idx]
                if self.flag:
                    print(2)
                    foldlabel_test, skeleton_test = foldlabel.copy(), skeleton.copy()
                    foldlabel_test[foldlabel_test>0] = 1
                    skeleton_test[skeleton_test>0] = 1
                    #if np.array_equal(foldlabel_test,skeleton_test):
                    assert(np.array_equal(foldlabel_test,skeleton_test))
                    #print(7)
                    self.flag =False
                angle = np.random.uniform(-3, 3)

            fill_value = 0
            print(3)
            mask = transforms.Compose([ApplyMask()])
            foldlabel = mask(foldlabel)
            aims.write(aims.Volume(np.squeeze(np.array(foldlabel))), f"/tmp/foldlabel_{idx}_mask.nii.gz")
            # List of simple surfaces
            print(4)
            folds_list = np.unique(foldlabel, return_counts=True)
            folds_dico = {key: value for key, value in zip(folds_list[0], folds_list[1]) if value>=300}
            #folds_dico.pop(0, None)
            distmap_masked_dict = {}

            for ss, ss_size in folds_dico.items():
                if ss!=0:
                    print(ss, ss_size)
                    inpaint = transforms.Compose([inferenceSuppression(foldlabel, ss)])
                    aims.write(dtx.convert.volume_to_bucketMap_aims(np.squeeze(np.array((mask(skeleton))))), f"/tmp/skel_{idx}_{ss}.bck")
                    skeleton_del, target = inpaint(skeleton)
                    #print(np.unique(foldlabel, return_counts=True))
                    aims.write(dtx.convert.volume_to_bucketMap_aims(np.squeeze(np.array(mask(skeleton_del)))), f"/tmp/skel_masked_{idx}_{ss}.bck")
                    distmap_masked = convert2Distmap(skeleton_del)

                    self.transform = transforms.Compose([
                                        NormalizeSkeleton(),
                                        ApplyMask(),
                                        Padding(list(self.config.in_shape),
                                                fill_value=fill_value)
                                        ])
                    distmap = np.squeeze(distmap)
                    distmap_masked = np.squeeze(distmap_masked)
                    distmap_masked_dict[ss_size] = self.transform(distmap_masked)

            tuple_with_path = (distmap_masked_dict,
                               self.transform(distmap),
                               filename)

            return tuple_with_path




class inferenceSuppression(object):
    """
    """
    def __init__(self, foldlabel, ss, min_size=100):
        foldlabel = np.squeeze(foldlabel, axis=0)
        self.foldlabel_del = np.copy(foldlabel)
        self.fold = ss
        self.min_size = min_size

    def __call__(self, skeleton):
        skeleton_del = np.copy(skeleton)
        target = np.zeros(list(skeleton_del.shape))
        assert(target.shape==skeleton_del.shape)

        # Selection of selected simple surface
        self.foldlabel_del[self.foldlabel_del==self.fold] = 9999
        # Selection of associated bottom
        self.foldlabel_del[self.foldlabel_del==self.fold + 6000] = 9999
        # Selection of other associated junction
        self.foldlabel_del[self.foldlabel_del==self.fold + 5000] = 9999

        ## suppression of chosen folds
        skeleton_del[self.foldlabel_del==9999] = -1
        assert(np.count_nonzero(skeleton_del==-1)>=300)
        #print(np.count_nonzero(skeleton==-1))
        skeleton_del[skeleton_del==-1] = 0
        #skeleton[skeleton==-1] = 1

        ## writing of deleted folds to reconstruct in target
        target[self.foldlabel_del==9999] = 1

        return skeleton_del, target
