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
Script in order to create benchmarks set
"""
import os
import sys

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from soma import aims

#cropped_dir = '/neurospin/dico/lguillon/inpainting/benchmark/asymmetry_v2/flip/Ldistmaps/L'
cropped_dir = '/neurospin/dico/lguillon/inpainting/benchmark/deletion_v2/1000/crops/1mm/SC/no_mask/Rlabels'
#tgt_dir = '/neurospin/dico/lguillon/inpainting/benchmark/asymmetry_v2/flip'
tgt_dir = '/neurospin/dico/lguillon/inpainting/benchmark/deletion_v2/1000/'
file_basename = 'Rlabels'

# Test subjects
test_list = pd.read_csv("/neurospin/dico/lguillon/distmap/data/test_list.csv")
test_list['subjects'] = test_list['subjects'].astype('str')
print(f"total of test subjects : {len(test_list)}")

list_sample_id = []
list_sample_file = []

for subject in test_list['subjects']:
    print(subject)
    file_nii = os.path.join(cropped_dir, subject+'_cropped_foldlabel.nii.gz')
    #file_nii = os.path.join(cropped_dir, 'Rdistmap_generated_'+subject+'.nii.gz')
    if '.minf' not in file_nii and os.path.exists(file_nii):
        aimsvol = aims.read(file_nii)
        sample = np.asarray(aimsvol)
        #subject = re.search('(.*)_cropped_(.*)', file_nii).group(1)
        list_sample_id.append(subject)
        list_sample_file.append(sample)

# Writes subject ID csv file
subject_df = pd.DataFrame(list_sample_id, columns=["Subject"])
subject_df.to_csv(os.path.join(tgt_dir, file_basename+'_subject.csv'),
                  index=False)
print(f"5 first saved subjects are: {subject_df.head()}")

# Writes subject ID to npy file (per retrocompatibility)
list_sample_id = np.array(list_sample_id)
np.save(os.path.join(tgt_dir, 'sub_id.npy'), list_sample_id)

# Writes volumes as numpy arrays
list_sample_file = np.array(list_sample_file)
np.save(os.path.join(tgt_dir, file_basename+'.npy'), list_sample_file)
