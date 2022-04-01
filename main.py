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
from train import train_model
from train_vae import train_vae
#from clustering import Cluster
from load_data import create_subset
from config import Config


#tb_logger = pl_loggers.TensorBoardLogger('logs')
#writer = SummaryWriter()


if __name__ == '__main__':

    config = Config()

    torch.manual_seed(0)
    save_dir = config.save_dir

    """ Load data and generate torch datasets """
    # if config.model == 'vae':
    #     subset1 = create_vae_subset(config)
    # else:
    subset1 = create_subset(config)
    train_set, val_set = torch.utils.data.random_split(subset1,
                            [round(0.8*len(subset1)), round(0.2*len(subset1))])
    trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=config.batch_size,
                  num_workers=1,
                  shuffle=True)
    valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=8,
                num_workers=8,
                shuffle=True)
    print(len(train_set), len(val_set))
    print(len(trainloader), len(valloader))

    #val_label = []
    #for _, path in valloader:
    #    val_label.append(path[0])
    #np.savetxt(f"{save_dir}val_label.csv", np.array(val_label), delimiter =", ", fmt ='% s')

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    weights = [1, config.weight]
    if config.model == 'vae':
        weights = [1, 2]
    #class_weights = torch.FloatTensor(weights).to(device)
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    criterion = torch.nn.MSELoss(reduction='sum')

    cur_config = { "n": config.n, "weight": weights}
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        print("Directory " , save_dir ,  " already exists")
        pass
    print(cur_config)

    """ Train model for given configuration """
    if config.model == 'vae':
        model, final_loss_val = train_vae(config, trainloader, valloader,
                                          root_dir=save_dir)
    else:
        model = train_model(config, trainloader, valloader, root_dir=save_dir)

    """ Evaluate model performances """
    """dico_set_loaders = {'train': trainloader, 'val': valloader}

    tester = ModelTester(model=vae, dico_set_loaders=dico_set_loaders,
                         kl_weight=config.kl, loss_func=criterion,
                         n_latent=config.n, depth=3)

    results = tester.test()
    encoded = {loader_name:[results[loader_name][k] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
    df_encoded = pd.DataFrame()
    df_encoded['latent'] = encoded['train'] + encoded['val']
    X = np.array(list(df_encoded['latent']))

    cluster = Cluster(X, save_dir)
    res = cluster.plot_silhouette()
    res['loss_val'] = final_loss_val

    with open(f"{save_dir}results_test.json", "w") as json_file:
        json_file.write(json.dumps(res, sort_keys=True, indent=4))"""
