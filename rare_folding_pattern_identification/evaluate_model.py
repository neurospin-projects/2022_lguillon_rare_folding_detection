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
import os
import sys

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

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
#from analyses.evaluate_model import anomaly_score, classifier
from load_data import create_subset, create_benchmark_subset
from config import Config

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score


def anomaly_score(X_train, X_benchmark):
    """
    """

    # Lsf: Anomaly score based on distance to OC-SVM
    clf = OneClassSVM(gamma='auto', nu=0.01).fit(X_train)
    Lsf = list(clf.decision_function(X_benchmark))

    # Lnn: anomaly score based on distance to 10 nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X_benchmark)
    distances, indices = nbrs.kneighbors(X_benchmark)
    Lnn = [np.mean(distances[k][1:]) for k in range(len(X_benchmark))]
    print(Lsf, Lnn)

    return Lsf, Lnn


def classifier(X_val, sub_train, X_benchmark, sub_bench):
    """
    """
    #X = np.concatenate((X_val[:100], X_benchmark))
    subjects_dict = {}
    print(len(X_val), len(X_benchmark))
    X = np.concatenate((X_val, X_benchmark))
    sub = np.concatenate((sub_train, sub_bench))
    label = np.array([0 for k in range(len(X_val))] + [1 for k in range(len(X_benchmark))])
    skf = StratifiedKFold(n_splits=3)
    av_log, av_svm, av_gb = [], [], []
    for train, test in skf.split(X, label):
        clf = LogisticRegression(random_state=0).fit(X[train], label[train])
        #av_log.append(clf.score(X[test], label[test]))
        pred = clf.predict(X[test])
        for i, k in enumerate(pred):
            if k != label[i]:
                if sub[i] not in subjects_dict.keys():
                    subjects_dict[sub[i]] = 1
        av_log.append(f1_score(label[test], pred, average='weighted'))

        svm = LinearSVC(random_state=0).fit(X[train], label[train])
        pred = svm.predict(X[test])
        for i, k in enumerate(pred):
            if k != label[i]:
                if sub[i] not in subjects_dict.keys():
                    subjects_dict[sub[i]] = 1
        av_svm.append(f1_score(label[test], pred, average='weighted'))

        gb = GradientBoostingClassifier(random_state=0).fit(X[train], label[train])
        pred = gb.predict(X[test])
        for i, k in enumerate(pred):
            if k != label[i]:
                if sub[i] not in subjects_dict.keys():
                    subjects_dict[sub[i]] = str(label[i])
        av_gb.append(f1_score(label[test], pred, average='weighted'))

    logreg, svm, gb = np.mean(av_log), np.mean(av_svm), np.mean(av_gb)
    print(logreg, svm, gb)
    return logreg, svm, gb, subjects_dict


def main():
    root_dir = f"/neurospin/dico/lguillon/distmap/"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)

    """ Evaluate model performances """
    #dico_set_loaders = {'ctrl': control_loader, 'amputee': amputee_loader, 'congenital': congenital_loader}
    train_set = create_subset(config, mode='evaluate')
    trainloader = torch.utils.data.DataLoader(
                                              train_set,
                                              batch_size=config.batch_size,
                                              num_workers=8,
                                              shuffle=True)
    benchmark_pre = create_benchmark_subset(config, config.benchmark_dir_1)
    benchmark_post = create_benchmark_subset(config, config.benchmark_dir_2)
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
    dico_set_loaders = {'train': trainloader,
                        'benchmark_pre': benchloader_pre,
                        'benchloader_post': benchloader_post}

    print(len(trainloader), len(benchloader_pre), len(benchloader_post))

    tester = ModelTester(model=vae, dico_set_loaders=dico_set_loaders,
                         kl_weight=kl, loss_func=criterion, n_latent=n,
                         depth=3)

    results = tester.test()
    encoded = {loader_name:[results[loader_name][k][1] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
    losses = {loader_name:[int(results[loader_name][k][0].cpu().detach().numpy()) for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}

    X_train = np.array(list(encoded['train']))
    sub_train = np.array(list(results['train'].keys()))

    """Precentral sulcus"""
    X_benchmark_pre = np.array(list(encoded['benchmark_pre']))
    sub_bench_pre = np.array(list(results['benchmark_pre'].keys()))

    res_clf = classifier(X_train, sub_train, X_benchmark_pre, sub_bench_pre)

    res['final_loss_val'] = final_loss_val
    res['logreg_pre'] = res_clf[0]
    res['svm_pre'] = res_clf[1]
    res['gb_pre'] = res_clf[2]
    print(res_clf[3])
    res['sub_problem_pre'] = res_clf[3]

    """Postcentral sulcus"""
    X_benchmark_post = np.array(list(encoded['benchloader_post']))
    sub_bench_post = np.array(list(results['benchloader_post'].keys()))
    res_clf = classifier(X_train, sub_train, X_benchmark_post, sub_bench_post)

    print('end of config')
    res['logreg_post'] = res_clf[0]
    res['svm_post'] = res_clf[1]
    res['gb_post'] = res_clf[2]
    print(res_clf[3])
    res['sub_problem_post'] = res_clf[3]

    with open(f"{root_dir}results_test.json", "w") as json_file:
        json_file.write(json.dumps(res, sort_keys=True, indent=4))

if __name__ == '__main__':
    main()
