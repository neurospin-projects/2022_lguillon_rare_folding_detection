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

"""

""" The aim of this script is to display saved model outputs thanks to Anatomist.
Model outputs are stored as numpy arrays.
"""

import anatomist.api as anatomist
from soma import aims
import numpy as np
import pandas as pd
import json


def array_to_ana(ana_a, sub_id, bck_dir):
    """
    Transforms output tensor into volume readable by Anatomist and defines
    name of the volume displayed in Anatomist window.
    Returns volume displayable by Anatomist
    """
    img = aims.read(f"{bck_dir}{sub_id}_cropped_distmap.bck")
    a_vol_img = ana_a.toAObject(img)
    img.header()['voxel_size'] = [1, 1, 1]
    a_vol_img.setName(str(sub_id)) # display name
    a_vol_img.setChanged()
    a_vol_img.notifyObservers()

    return img, a_vol_img


def main():
    """
    In the Anatomist window, for each model output, corresponding input will
    also be displayed at its left side.
    Number of columns and view (Sagittal, coronal, frontal) can be specified.
    (It's better to choose an even number for number of columns to display)
    """
    root_dir = "/neurospin/dico/lguillon/distmap/pre_post_ambiguity_search/"
    bck_dir = "/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/postcentral/no_mask/benchmark/Rbuckets/"
    #bck_dir = "/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rbuckets/"

    #sub_dir = f"{root_dir}results_test.json"
    sub_dir = f"{root_dir}post_ambiguity.json"
    with open(sub_dir, 'r') as f:
        data = json.load(f)
    #sub_list = list(data['sub_problem_pre'].keys())
    sub_list = list(data.keys())
    n_sub = len(sub_list)

    a = anatomist.Anatomist()
    block = a.AWindowsBlock(a, 6)  # Parameter 6 corresponds to the number of columns displayed. Can be changed.

    for k in range(n_sub):
        sub_id = sub_list[k]
        #if data['sub_problem_pre'][sub_id]=='1':
        if True:
            globals()['block%s' % (sub_id)] = a.createWindow('3D', block=block)
            globals()['img%s' % (sub_id)], globals()['a_img%s' % (sub_id)] = array_to_ana(a, sub_id, bck_dir)
            globals()['block%s' % (sub_id)].addObjects(globals()['a_img%s' % (sub_id)])



if __name__ == '__main__':
    main()
    from soma.qt_gui.qt_backend import Qt
    Qt.qApp.exec_()
