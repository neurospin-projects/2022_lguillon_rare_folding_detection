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


def classifier(X_val, X_benchmark):
    """
    """
    np.random.shuffle(X_val)
    #X = np.concatenate((X_val[:100], X_benchmark))
    print(len(X_val), len(X_benchmark))
    X = np.concatenate((X_val, X_benchmark))
    label = np.array([0 for k in range(len(X_val))] + [1 for k in range(len(X_benchmark))])
    skf = StratifiedKFold(n_splits=3)
    av_log, av_svm, av_gb = [], [], []
    for train, test in skf.split(X, label):
        clf = LogisticRegression(random_state=0).fit(X[train], label[train])
        #av_log.append(clf.score(X[test], label[test]))
        pred = clf.predict(X[test])
        av_log.append(f1_score(label[test], pred, average='weighted'))

        svm = LinearSVC(random_state=0).fit(X[train], label[train])
        pred = svm.predict(X[test])
        av_svm.append(f1_score(label[test], pred, average='weighted'))

        gb = GradientBoostingClassifier(random_state=0).fit(X[train], label[train])
        pred = gb.predict(X[test])
        av_gb.append(f1_score(label[test], pred, average='weighted'))

    logreg, svm, gb = np.mean(av_log), np.mean(av_svm), np.mean(av_gb)
    print(logreg, svm, gb)
    return logreg, svm, gb
