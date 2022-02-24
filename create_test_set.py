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
import random
from config import Config


config = Config()

# Right handed subjects (that might not have been processed with Morphologist)
train_list = pd.read_csv("/neurospin/dico/lguillon/hcp_info/right_handed_dataset.csv")
train_list['subjects'] = train_list['subjects'].astype('str')
print(f"total of right_handed subjects : {len(train_list)}")

# subjects of HCP indeed processed with Morphologist
hcp_sub = pd.DataFrame(os.listdir('/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist/'),
                           columns=["subjects"])
train_list = train_list.merge(hcp_sub, left_on='subjects', right_on='subjects')
print(f"total of right handed subjects processed with Morphologist : {len(train_list)}")

# random selection of 200 subjects
test_list = random.sample(list(train_list.subjects), 200)
train_list = list(set(list(train_list.subjects))-set(test_list))
benchmark_list = random.sample(train_list, 100)
train_list = list(set(train_list)-set(benchmark_list))
print(f"total subjects for training : {len(train_list)}")
print(f"total subjects for testing : {len(test_list)}")
print(f"total subjects for testing : {len(benchmark_list)}")

# saving of subjects lists
train_list = pd.DataFrame(train_list, columns=['subjects'])
test_list = pd.DataFrame(test_list, columns=['subjects'])
benchmark_list = pd.DataFrame(benchmark_list, columns=['subjects'])
train_list.to_csv('/neurospin/dico/lguillon/miccai_22/data/train_list_2.csv')
test_list.to_csv('/neurospin/dico/lguillon/miccai_22/data/test_list_2.csv')
benchmark_list.to_csv('/neurospin/dico/lguillon/miccai_22/data/benchmark_list_2.csv')
