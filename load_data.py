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

"""
Tools in order to create pytorch dataloaders
"""
import os
import sys

import pandas as pd
import numpy as np
from preprocess import *
from config import Config


def create_subset(config):
    """
    Creates dataset from HCP data

    Args:
        config: instance of class Config

    Returns:
        subset: Dataset corresponding to HCP
    """
    ######## TO CHANGE ########
    train_list = pd.read_csv(config.subject_dir, header=None, usecols=[0],
                             names=['subjects'])
    train_list['subjects'] = train_list['subjects'].astype('str')

    skeletons = pd.read_pickle(os.path.join(config.data_dir, "Rskeleton.pkl")).T
    skeletons.index.astype('str')

    skeletons = skeletons.rename(columns={0: "skeleton"})

    foldlabels = pd.read_pickle(os.path.join(config.data_dir, "Rlabels.pkl")).T
    foldlabels.index.astype('str')
    foldlabels = foldlabels.rename(columns={0: "labels"})
    foldlabels['subjects'] = foldlabels.index.astype('str')

    skeletons = skeletons.merge(train_list, left_on = skeletons.index, right_on='subjects', how='right')
    foldlabels = foldlabels.merge(skeletons, left_on = foldlabels.index, right_on='subjects', how='right')

    filenames = list(train_list['subjects'])

    subset = SkeletonDataset(dataframe=foldlabels.head(), filenames=filenames,
                             min_size=config.min_size, visu_check=True)

    return subset


def main():
    config = Config()
    subset = create_subset(config)

    trainloader = torch.utils.data.DataLoader(
                  subset,
                  batch_size=config.batch_size,
                  num_workers=8,
                  shuffle=True)
    input_arr = []
    output_arr = []
    id_arr = []
    for (sample, path), orig_sample in trainloader:
        print(path)
        print(np.unique(sample))
        print(sample.shape)
        for k in range(len(path)):
            input_arr.append(np.array(np.squeeze(orig_sample[k]).cpu().detach().numpy()))
            output_arr.append(np.array(np.squeeze(sample[k]).cpu().detach().numpy()))
            id_arr.append(path[k])

    np.save(config.save_dir+'input.npy', np.array([input_arr]))
    np.save(config.save_dir+'output.npy', np.array([output_arr]))
    np.save(config.save_dir+'id.npy', np.array([id_arr]))


if __name__ == '__main__':
    main()
