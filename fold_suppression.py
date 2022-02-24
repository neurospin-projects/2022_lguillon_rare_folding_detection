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
Random fold suppression module
"""

import pandas as pd
import numpy as np
import random


def random_suppression(sample, foldlabel_map):
    """
    Randomly deletes folds from a foldlabel_map until a total of x voxels are
    suppressed

    Args:
        sample: numpy array of an image where to delete folds
        foldlabel_map: volume containing folds labels

    Returns:
        sample: sample with folds deleted
    """
    del_list = []
    folds_list = np.unique(foldlabel_map, return_counts=True)
    folds_dico = {key: value for key, value in zip(folds_list[0], folds_list[1])}

    # We don't take into account the background in the random choice of fold
    folds_dico.pop(0, None)

    # Random choice of fold
    fold = random.choice(list(folds_dico.keys()))

    # if fold size < 100, deletion of other folds
    if folds_dico[fold]<100:
        total_del = folds_dico[fold]
        del_list.append(fold)
        folds_dico.pop(fold, None)
        while total_del <= 100:
            fold = random.choice([k for k, v in folds_dico.items() if v<100])
            total_del += folds_dico[fold]
            del_list.append(fold)
            folds_dico.pop(fold, None)
    # if fold size >= 100, suppression of this fold only
    else:
        del_list.append(fold)

    print(del_list)

    # Suppression of equivalent voxels in skeleton
    ## Marking of folds to delete on foldlabel_map
    for fold in del_list:
        foldlabel_map[foldlabel_map==fold] = 9999

    print(np.unique(foldlabel_map))

    ## suppression of chosen folds
    print(np.unique(sample, return_counts=True))
    sample[foldlabel_map==9999] = 0
    print(np.unique(sample, return_counts=True))
    return sample


def main():
    labels_src_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
                     'mask/sulcus_based/2mm/Rlabels.pkl'
    skeleton_src_dir = '/neurospin/dico/data/deep_folding/current/crops/SC/' \
                       'mask/sulcus_based/2mm/Rskeleton.pkl'

    labels = pd.read_pickle(labels_src_dir)
    skeleton = pd.read_pickle(skeleton_src_dir)

    # labels
    column = labels.columns[0]
    print(column)
    foldlabel = labels[column][0]

    # skeletons
    sample = skeleton[column][0]

    random_suppression(sample, foldlabel)

    #print(foldlabel)


if __name__ == '__main__':
    main()
