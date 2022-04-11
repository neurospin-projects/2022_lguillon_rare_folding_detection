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
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from vae import ModelTester
#from preprocess import AugDatasetTransformer
from train_vae import train_vae
from analyses.evaluate_model import anomaly_score, classifier
from load_data import create_subset, create_benchmark_subset
from config import Config


def gridsearch_bVAE_sub1(subset, benchmark_pre, benchmark_post):
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

    criterion = torch.nn.MSELoss(reduction='sum')

    grid_config = {"kl": [1, 2, 4, 8, 10],
              "n": [2, 4, 10, 20, 40, 75, 100]
    }

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
    """control_loader = torch.utils.data.DataLoader(control_dataset, batch_size=1,
                                                    shuffle=True, num_workers=8)
    amputee_loader = torch.utils.data.DataLoader(amputee_dataset, batch_size=1,
                                                    shuffle=True, num_workers=8)
    congenital_loader = torch.utils.data.DataLoader(congenital_dataset, batch_size=1,
                                                    shuffle=True, num_workers=8)"""


    for kl, n in list(itertools.product(grid_config["kl"], grid_config["n"])):
        cur_config = {"kl": kl, "n": n}
        root_dir = f"/neurospin/dico/lguillon/distmap/gridsearch/n_{n}_kl_{kl}/"
        #root_dir = config.save_dir
        res = {}

        try:
            os.mkdir(root_dir)
        except FileExistsError:
            print("Directory " , root_dir ,  " already exists")
            pass
        print(cur_config)
        #k= 0
        #kfold = KFold(n_splits=3, shuffle=True)
        #for fold, (train_ids, test_ids) in enumerate(kfold.split(subset)):
        #    print(k)
        #fold_Lbvae, fold_Lsf, fold_Lnn = [], [], []
        fold_logreg, fold_svm, fold_gb = [], [], []
        train_set, val_set = torch.utils.data.random_split(subset,
                            [round(0.8*len(subset)), round(0.2*len(subset))])

        #train_set = AugDatasetTransformer(train_set)
        #print(1)
        #val_set = AugDatasetTransformer(val_set, aug=False)

        trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=config.batch_size,
                  num_workers=8,
                  shuffle=True)
        valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=64,
                num_workers=8,
                shuffle=True)

        print('size of train loader: ', len(trainloader), 'size of val loader: ',
              len(valloader))

        """ Train model for given configuration """
        vae, final_loss_val = train_vae(config, trainloader, valloader,
                                        root_dir=root_dir,
                                        curr_config=cur_config)

        """ Evaluate model performances """
        #dico_set_loaders = {'ctrl': control_loader, 'amputee': amputee_loader, 'congenital': congenital_loader}
        train_set
        dico_set_loaders = {'train': trainloader,
                            'benchmark_pre': benchloader_pre,
                            'benchloader_post': benchloader_post}

        tester = ModelTester(model=vae, dico_set_loaders=dico_set_loaders,
                             kl_weight=kl, loss_func=criterion, n_latent=n,
                             depth=3)

        results = tester.test()
        encoded = {loader_name:[results[loader_name][k][1] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
        losses = {loader_name:[int(results[loader_name][k][0].cpu().detach().numpy()) for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
        X_train = np.array(list(encoded['train']))
        #X_control = np.array(list(encoded['ctrl']) + list(encoded['amputee']) )
        #X_oh = np.array(list(encoded['congenital']))
        X_benchmark_pre = np.array(list(encoded['benchmark_pre']))
        sub_train = np.array(list(results['train'].keys()))
        sub_bench_pre = np.array(list(results['benchmark_pre'].keys()))
        res_clf = classifier(X_train, sub_train, X_benchmark_pre, sub_bench_pre)
        #res_clf = classifier(X_control, X_oh)

        #Lsf, Lnn = anomaly_score(X_train, X_benchmark)
        #Lbvae = np.mean(losses['benchmark'])

        #fold_Lbvae.append(Lbvae)
        #fold_Lsf.append(Lsf)
        #fold_Lnn.append(Lnn)
        fold_logreg.append(res_clf[0])
        fold_svm.append(res_clf[1])
        fold_gb.append(res_clf[2])

        print('end of config')
        res['final_loss_val'] = final_loss_val
        res['logreg'] = np.mean(fold_logreg)
        res['svm'] = np.mean(fold_svm)
        res['gb'] = np.mean(fold_gb)
        print(res_clf[3])
        res['sub_problem'] = res_clf[3]

        with open(f"{root_dir}results_test.json", "w") as json_file:
            json_file.write(json.dumps(res, sort_keys=True, indent=4))


def main():
    """ Main function to perform gridsearch on betaVAE
    """
    torch.manual_seed(0)
    config = Config()
    #root_dir = f"/neurospin/dico/lguillon/midl_22/new_design/gridsearch/"

    """ Load data and generate torch datasets """
    subset = create_subset(config)

    benchmark_pre = create_benchmark_subset(config, config.benchmark_dir_1)
    benchmark_post = create_benchmark_subset(config, config.benchmark_dir_2)
    #control_dataset, amputee_dataset, congenital_dataset = create_one_handed_subset(config)

    gridsearch_bVAE_sub1(subset, benchmark_pre, benchmark_post)


if __name__ == '__main__':
    main()
