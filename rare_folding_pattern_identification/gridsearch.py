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

# p = os.path.abspath('../')
# if p not in sys.path:
#     sys.path.append(p)

import numpy as np
import pandas as pd
import json
import itertools
import torch
import random
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from vae import ModelTester
#from preprocess import AugDatasetTransformer
from train_vae import train_vae
from analyses.evaluate_model import anomaly_score, classifier
from load_data import create_subset, create_benchmark_subset
from config import Config


def gridsearch_bVAE_sub1():
    """ Applies a gridsearch to find best hyperparameters configuration (beta
    value=kl and latent space size=n) based on loss value, silhouette score and
    reconstruction abilities

    Args:
        trainloader: torch loader of training data
        valloader: torch loader of validation data
    """
    config = Config()
    torch.manual_seed(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)

    """ Load data and generate torch datasets """
    train_set, val_set = create_subset(config, mode='train')

    criterion = torch.nn.MSELoss(reduction='sum')

    grid_config = {"kl": [2, 4, 8, 10],
              "n": [4, 20, 40, 50, 75, 100, 150]
    }

    """control_loader = torch.utils.data.DataLoader(control_dataset, batch_size=1,
                                                    shuffle=True, num_workers=8)
    amputee_loader = torch.utils.data.DataLoader(amputee_dataset, batch_size=1,
                                                    shuffle=True, num_workers=8)
    congenital_loader = torch.utils.data.DataLoader(congenital_dataset, batch_size=1,
                                                    shuffle=True, num_workers=8)"""


    for kl, n in list(itertools.product(grid_config["kl"], grid_config["n"])):
        cur_config = {"kl": kl, "n": n}
        config.kl, config.n = kl, n
        root_dir = f"/neurospin/dico/lguillon/distmap/gridsearch_lr5e-4/n_{n}_kl_{kl}/"
        #root_dir = config.save_dir
        res = {}

        try:
            os.mkdir(root_dir)
        except FileExistsError:
            print("Directory " , root_dir ,  " already exists")
            pass
        print(cur_config)

        trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=config.batch_size,
                  num_workers=8,
                  shuffle=True)
        valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=config.batch_size,
                num_workers=8,
                shuffle=True)

        print('size of train loader: ', len(trainloader), 'size of val loader: ',
              len(valloader))

        """ Train model for given configuration """
        vae, final_loss_val = train_vae(config, trainloader, valloader,
                                        root_dir=root_dir,
                                        curr_config=config)

        """ Evaluate model performances """
        #dico_set_loaders = {'ctrl': control_loader, 'amputee': amputee_loader, 'congenital': congenital_loader}
        indices = random.sample(range(0, len(val_set)), 46)
        print(indices)
        val_subset = torch.utils.data.Subset(val_set, indices=indices)
        valloader = torch.utils.data.DataLoader(val_subset,
                                                batch_size=config.batch_size,
                                                num_workers=8,
                                                shuffle=False)
        benchmark_pre = create_benchmark_subset(config, config.benchmark_dir_1,
                                                gridsearch=True, bench='pre')
        benchmark_post = create_benchmark_subset(config, config.benchmark_dir_2,
                                                 gridsearch=True, bench='post')
        benchloader_pre = torch.utils.data.DataLoader(
                          benchmark_pre,
                          batch_size=config.batch_size,
                          num_workers=8,
                          shuffle=False)
        benchloader_post = torch.utils.data.DataLoader(
                           benchmark_post,
                           batch_size=config.batch_size,
                           num_workers=8,
                           shuffle=False)
        dico_set_loaders = {'val': valloader,
                            'benchmark_pre': benchloader_pre,
                            'benchmark_post': benchloader_post}

        print(len(valloader), len(benchloader_pre), len(benchloader_post))

        tester = ModelTester(model=vae, dico_set_loaders=dico_set_loaders,
                             kl_weight=kl, loss_func=criterion, n_latent=n,
                             depth=3)

        results = tester.test()
        encoded = {loader_name:[results[loader_name][k][1] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
        losses = {loader_name:[int(results[loader_name][k][0].cpu().detach().numpy()) for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}

        X_sc = np.array(list(encoded['val']))
        sub_sc = np.array(list(results['val'].keys()))

        """Pre/post central sulcus"""
        X_benchmark = np.array(list(encoded['benchmark_pre']) + list(encoded['benchmark_post']))
        sub_bench = np.array(list(results['benchmark_pre'].keys()) + list(results['benchmark_post'].keys()))

        res_clf = classifier(X_sc, sub_sc, X_benchmark, sub_bench)

        res['final_loss_val'] = final_loss_val
        res['logreg_pre'] = res_clf[0]
        res['svm_pre'] = res_clf[1]
        res['gb_pre'] = res_clf[2]
        print(res_clf[3])
        res['sub_problem_pre'] = res_clf[3]

        with open(f"{root_dir}results_test.json", "w") as json_file:
            json_file.write(json.dumps(res, sort_keys=True, indent=4))


def main():
    """ Main function to perform gridsearch on betaVAE
    """
    gridsearch_bVAE_sub1()


if __name__ == '__main__':
    main()
