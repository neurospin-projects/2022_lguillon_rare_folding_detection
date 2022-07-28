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
import csv
import random
from preprocess import *
from config import Config


def create_subset(config, mode):
    """
    Creates dataset from HCP data

    Args:
        config: instance of class Config

    Returns:
        subset: Dataset corresponding to HCP
    """
    np.random.seed(1)

    filenames = np.load(os.path.join(config.data_dir,
                                    "train_sub_id.npy"))
    distmaps = np.load(os.path.join(config.data_dir,
                                    "train_distmap.npy"),
                       mmap_mode='r')

    indices = list(range(len(filenames)))
    np.random.shuffle(indices)
    split = int(np.floor(0.8 * len(filenames)))
    train_idx, val_idx = indices[:split], indices[split:]
    train_distmap, train_filenames = distmaps[train_idx], filenames[train_idx]
    val_distmap, val_filenames = distmaps[val_idx], filenames[val_idx]

    if mode=='train':
        train_set = SkeletonDataset(dataframe=train_distmap,
                                    filenames=train_filenames,
                                    data_transforms=True)
        val_set = SkeletonDataset(dataframe=val_distmap,
                                  filenames=val_filenames,
                                  data_transforms=False)
        return train_set, val_set

    else:
        train_set = SkeletonDataset(dataframe=train_distmap,
                                    filenames=train_filenames,
                                    data_transforms=False)

        return train_set


def create_inpaint_subset(config, mode='train'):
    """
    Creates dataset from HCP data for inpainting model

    Args:
        config: instance of class Config

    Returns:
        subset: Dataset corresponding to HCP
    """
    np.random.seed(1)

    filenames_ref = np.load(os.path.join(config.train_idx_dir,
                                    "train_sub_id.npy"))

    distmaps = np.load(os.path.join(config.data_dir, "Rskel_distmaps",
                                    "skel_distmap_1mm.npy"),
                       mmap_mode='r')

    skeletons = np.load(os.path.join(config.data_dir, "Rskeletons",
                                    "skeletons_1mm.npy"),
                       mmap_mode='r')
    filenames = np.load(os.path.join(config.data_dir,"Rfoldlabels",
                                    "sub_id.npy"))
    foldlabels = np.load(os.path.join(config.data_dir, "Rfoldlabels",
                                    "foldlabels_1mm.npy"),
                       mmap_mode='r')
    print('ici')
    sorter = np.argsort(filenames)
    filenames_idx = sorter[np.searchsorted(filenames, filenames_ref, sorter=sorter)]
    filenames = filenames[filenames_idx]
    distmaps = distmaps[filenames_idx]
    skeletons = skeletons[filenames_idx]
    foldlabels = foldlabels[filenames_idx]

    indices = list(range(len(filenames_ref)))
    np.random.shuffle(indices)
    split = int(np.floor(0.8 * len(filenames)))
    train_idx, val_idx = indices[:split], indices[split:]

    train_distmaps, train_filenames = distmaps[train_idx], filenames[train_idx]
    train_skeletons, train_foldlabels = skeletons[train_idx], foldlabels[train_idx]
    val_distmaps, val_filenames = distmaps[val_idx], filenames[val_idx]
    val_skeletons, val_foldlabels = skeletons[val_idx], foldlabels[val_idx]
    print('la')

    #data = np.array([foldlabels, skeletons, distmaps])

    # indices = list(range(len(filenames)))
    # np.random.shuffle(indices)
    # split = int(np.floor(0.8 * len(filenames)))
    # train_idx, val_idx = indices[:split], indices[split:]
    # train_distmap, train_filenames = distmaps[train_idx], filenames[train_idx]
    # val_distmap, val_filenames = distmaps[val_idx], filenames[val_idx]

    if mode=='train':
        train_set = InpaintDataset(foldlabels=train_foldlabels,
                                    skeletons = train_skeletons,
                                    distmaps = train_distmaps,
                                    filenames=train_filenames,
                                    data_transforms=True)
        val_set = InpaintDataset(foldlabels=val_foldlabels,
                                 skeletons = val_skeletons,
                                 distmaps = val_distmaps,
                                 filenames=val_filenames,
                                  data_transforms=True)
        return train_set, val_set

    else:
        train_set = InpaintDataset(foldlabels=foldlabels,
                                    filenames=filenames,
                                    skeletons = skeletons,
                                    distmaps = distmaps,
                                    data_transforms=False)
        return train_set


def create_benchmark_subset(config, benchmark_dir, gridsearch=False,
                            bench=False):
    """
    Creates dataset from benchmark data to identify ambiguous subjects
    Benchmark is composed of crops of precentral and postcentral sulci

    Args:
        config: instance of class Config

    Returns:
        subset: Dataset corresponding to benchmark data
    """
    if gridsearch:
        if bench=='pre':
            df = pd.read_csv("/neurospin/dico/lguillon/distmap/" \
                             "pre_post_ambiguity_search/subject_pre.csv")
        elif bench=='post':
            df = pd.read_csv("/neurospin/dico/lguillon/distmap/" \
                             "pre_post_ambiguity_search/subject_post.csv")
    else:
        df = pd.read_csv(config.subject_dir)

    train_list = np.array(list(df.subjects))
    distmaps = np.load(os.path.join(benchmark_dir, "distmap_1mm.npy"),
                       mmap_mode='r')
    filenames = np.load(os.path.join(config.data_dir, "train_sub_id.npy"))

    sorter = np.argsort(filenames)
    filenames_idx = sorter[np.searchsorted(filenames, train_list, sorter=sorter)]
    filenames = filenames[filenames_idx]
    distmaps = distmaps[filenames_idx]

    print(distmaps.shape, filenames.shape)

    subset = SkeletonDataset(dataframe=distmaps,
                             filenames=filenames,
                             data_transforms=False)

    return subset


def create_inpaint_benchmark_subset(config, benchmark_dir, gridsearch=False,
                            bench=False):
    """
    Creates dataset from benchmark data to identify ambiguous subjects
    Benchmark is composed of crops of precentral and postcentral sulci

    Args:
        config: instance of class Config

    Returns:
        subset: Dataset corresponding to benchmark data
    """
    if gridsearch:
        if bench=='pre':
            df = pd.read_csv("/neurospin/dico/lguillon/distmap/" \
                             "pre_post_ambiguity_search/subject_pre.csv")
        elif bench=='post':
            df = pd.read_csv("/neurospin/dico/lguillon/distmap/" \
                             "pre_post_ambiguity_search/subject_post.csv")
    else:
        df = pd.read_csv(config.subject_dir)

    train_list = np.array(list(df.subjects))

    filenames = np.load('/neurospin/dico/lguillon/distmap/data/train_sub_id.npy')


    distmaps = np.load(os.path.join(benchmark_dir, "skel_distmap_1mm.npy"),
                       mmap_mode='r')

    skeletons = np.load(os.path.join(benchmark_dir, "Rskeletons",
                                    "skeletons_1mm.npy"),
                       mmap_mode='r')
    # filenames = np.load(os.path.join(benchmark_dir,"Rfoldlabels",
    #                                 "sub_id.npy"))
    foldlabels = np.load(os.path.join(benchmark_dir, "foldlabels_1mm.npy"),
                       mmap_mode='r')

    sorter = np.argsort(filenames)
    filenames_idx = sorter[np.searchsorted(filenames, train_list, sorter=sorter)]
    filenames = filenames[filenames_idx]
    distmaps = distmaps[filenames_idx]
    foldlabels = foldlabels[filenames_idx]
    skeletons = skeletons[filenames_idx]

    print(distmaps.shape, filenames.shape)

    subset = InpaintDataset(foldlabels=foldlabels,
                            filenames=filenames,
                            skeletons = skeletons,
                            distmaps = distmaps,
                            data_transforms=False)
    return subset


def create_one_handed_subset(config):
    """
    Creates dataset from benchmark data: crops of precentral and postcentral
    sulci

    Args:
        config: instance of class Config

    Returns:
        subset: Dataset corresponding to benchmark data
    """
    ######## TO CHANGE ########
    oh_dir = config.one_handed_dir

    labels = pd.read_csv('/neurospin/dico/lguillon/ohbm_22/one_handed_labels.csv')
    ctrl = labels[labels['Dominant hand']=='R']
    one_handed = labels[labels['Group']!='CTR']
    one_handed = labels[labels['Amp. Side']=='L']
    amputee = one_handed[one_handed['Group']=='AMP']
    cong = one_handed[one_handed['Group']=='CONG']

    data_dir = "/neurospin/dico/data/deep_folding/current/crops/SC/mask/sulcus_based/2mm/one_handed_dataset/"

    tmp3 = pd.read_pickle(oh_dir+'R_one_handed_skeleton.pkl')
    tmp3 = tmp3.T
    tmp3 = tmp3.rename(columns={0:'skeleton'})
    tmp3['subjects'] = [list(tmp3.index)[k][0:4] for k in range(len(tmp3))]

    controls = tmp3.merge(ctrl, left_on=tmp3.subjects, right_on='SubjID', how='inner')
    filenames = list(controls.SubjID)
    control_dataset = SkeletonDataset(dataframe=controls, filenames=filenames, visu_check=False)

    amputee = tmp3.merge(amputee, left_on=tmp3.subjects, right_on='SubjID', how='inner')
    filenames_amp = list(amputee.SubjID)
    amputee_dataset = SkeletonDataset(dataframe=amputee, filenames=filenames_amp,visu_check=False)

    congenital = tmp3.merge(cong, left_on=tmp3.subjects, right_on='SubjID', how='inner')
    filenames_cong = list(cong.SubjID)
    congenital_dataset = SkeletonDataset(dataframe=congenital, filenames=filenames_cong, visu_check=False)

    return control_dataset, amputee_dataset, congenital_dataset


def main():
    config = Config()
    #subset = create_subset(config)

    train, val = create_inpaint_subset(config)
    print('ok !')

    trainloader = torch.utils.data.DataLoader(
                  train,
                  batch_size=1,
                  num_workers=1,
                  shuffle=False)
    input_arr = []
    output_arr = []
    target_arr = []
    id_arr = []
    for distmap_masked, distmap, path in trainloader:
        print(path)
        print(distmap_masked.shape)
        #for k in range(len(path)):
            #input_arr.append(np.array(np.squeeze(sample[k]).cpu().detach().numpy()))
            #output_arr.append(np.array(np.squeeze(sample[k]).cpu().detach().numpy()))
            #target_arr.append(np.array(np.squeeze(target[k]).cpu().detach().numpy()))
            #id_arr.append(path[k])

    """np.save(config.save_dir+'input.npy', np.array([input_arr]))
    #np.save(config.save_dir+'output.npy', np.array([output_arr]))
    #np.save(config.save_dir+'target.npy', np.array([target_arr]))
    np.save(config.save_dir+'id.npy', np.array([id_arr]))"""


if __name__ == '__main__':
    main()
