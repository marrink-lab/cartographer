#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 University of Groningen

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path

import networkx as nx
import numpy as np
from sklearn import model_selection, svm, preprocessing, pipeline, base
from sklearn.externals import joblib

from ecfp import XCFPFingerprinter
from parse_db import parse_db


def invariant(graph, node_key):
    hetero_neighbors = [neighbor for neighbor in graph[node_key]
                        if graph.nodes[neighbor]['element'] not in 'CH']
    halogens = 'F Cl Br I'.split()
    halogen_neighbors = [neighbor for neighbor in graph[node_key]
                        if graph.nodes[neighbor]['element'] in halogens]
    cycles = nx.cycle_basis(graph)
    my_cycles = []
    for cycle in cycles:
        if node_key in cycle:
            my_cycles.append(cycle)
    if my_cycles:
        cycle_invar = len(min(my_cycles, key=len))
    else:
        cycle_invar = 0
    invariant = tuple((len(graph[node_key]),  # number of neighbours
                       len(hetero_neighbors),
                       len(halogen_neighbors),
                       graph.nodes[node_key]['hcount'],
                       graph.nodes[node_key]['element'] not in 'CH',
                       graph.nodes[node_key]['charge'],
                       cycle_invar))
    return invariant


class RounderMixIn:
    def predict(self, *args, **kwargs):
            predicted = super().predict(*args, **kwargs)
            return self.do_round(predicted)

    @staticmethod
    def do_round(y):
        return y.round().astype(int)


def draw_mol(mol):
    labels = nx.get_node_attributes(mol, 'name')
    nx.draw_networkx(mol, labels=labels)


class SVRR(RounderMixIn, svm.SVR):
    pass


def featurize(mol, feature_size=7, fingerprint_size=2):
    fingerprinter = XCFPFingerprinter(fingerprint_size, invariant=invariant)
    fp = fingerprinter.fingerprint(mol)
    feat_arr = np.zeros(feature_size, dtype=int)
    for val, count in fp.items():
        feat_arr[val % NUM_FEATURES] += count
    return feat_arr.reshape(1, -1)


class FingerPrinter(base.TransformerMixin):
    def __init__(self, feature_size, fp_radius):
        self.feature_size = feature_size
        self.fp_radius = fp_radius

    def get_params(self, deep=False):
        return {'feature_size': self.feature_size,
                'fp_radius': self.fp_radius}

    def set_params(self, **vals):
        for name, val in vals.items():
            setattr(self, name, val)

    def transform(self, X):
        if isinstance(X, nx.Graph):
            return featurize(X, self.feature_size, self.fp_radius)
        features = []
        for mol in X:
            features.append(featurize(mol, self.feature_size, self.fp_radius))
        return np.reshape(features, (-1, self.feature_size))

    def fit(self, X, y=None):
        return self


NUM_FEATURES = 7
FINGERPRINT_SIZE = 3
FILENAME = 'numbead_predictor.gz'
BASE_PATH = '/home/.../Documents/database'
XLS_FILE = os.path.join(BASE_PATH, 'DRUGS-06.xlsx')
AA_DIR = os.path.join(BASE_PATH, 'atomistic')
CG_DIR = os.path.join(BASE_PATH, 'Martini')
MAP_DIR = os.path.join(BASE_PATH, 'mapping')


scaler = preprocessing.RobustScaler()
finger_printer = FingerPrinter(NUM_FEATURES, FINGERPRINT_SIZE)
# Hyperparameters are determined by hyperopt_num_beads.py
svrr = SVRR()
params = dict((('estimator__C', 825.404185268019),
   ('estimator__epsilon', 0.05623413251903491),
   ('estimator__kernel', 'rbf'),
   ('estimator__gamma', 0.0031622776601683794),
   ('estimator__shrinking', True)))

MODEL = pipeline.Pipeline(steps=[('fingerprint', finger_printer),
                                 ('scale', scaler),
                                 ('estimator', svrr)])
MODEL.set_params(**params)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    molecules = parse_db(XLS_FILE, AA_DIR, CG_DIR, MAP_DIR)

    results = []
    aa_mols = []
    for _, aa_mol, cg_mol, _, _ in molecules:
        results.append(len(cg_mol))
        aa_mols.append(aa_mol)

    splits = model_selection.train_test_split(aa_mols, results, test_size=0.2)
    trainX, testX, trainY, testY = splits
    train = trainX, trainY
    test = testX, testY

    cv_out = model_selection.cross_validate(MODEL, *train, cv=4)
    print('Cross validation scores on the training set:')
    print(cv_out['test_score'])

    MODEL.fit(*train)

    joblib.dump(MODEL, FILENAME, compress=True)

    print('Score for the validation set:')
    print(MODEL.score(*test))

    plt.scatter(train[1], MODEL.predict(train[0]), c='b', label='train')
    plt.scatter(test[1], MODEL.predict(test[0]), c='r', label='test')
    xmin, xmax = plt.xlim()

    plt.plot([xmin, xmax], [xmin, xmax], '--')
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.show()

    predicted = MODEL.predict(aa_mols)
    diff = np.abs(results - predicted)
    print('Differences between expected and predicted:')
    print(diff)
