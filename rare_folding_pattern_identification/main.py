# /usr/bin/env python3
# coding: utf-8
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
import sys

import numpy as np
import pandas as pd
import json
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

#from model import ModelTester
#from train import train_model
from train_vae import train_vae
#from clustering import Cluster
from load_data import create_subset
from config import Config



if __name__ == '__main__':

    config = Config()

    torch.manual_seed(10)
    save_dir = config.save_dir

    """ Load data and generate torch datasets """
    train_set, val_set = create_subset(config, mode='train')
    trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=8,
                  num_workers=8,
                  shuffle=True)
    valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=8,
                num_workers=8,
                shuffle=True)
    print(len(train_set), len(val_set))
    print(len(trainloader), len(valloader))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    try:
        os.mkdir(save_dir)
    except FileExistsError:
        print("Directory " , save_dir ,  " already exists")
        pass

    """ Train model for given configuration """
    if config.model == 'vae':
        model, final_loss_val = train_vae(config, trainloader, valloader,
                                          root_dir=save_dir)
    else:
        model = train_model(config, trainloader, valloader, root_dir=save_dir)

