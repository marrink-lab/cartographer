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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn import model_selection, ensemble, preprocessing, pipeline
from sklearn.externals import joblib

from ecfp import XCFPFingerprinter
from parse_db import parse_db


def invariant(graph, node_key):
    hetero_neighbors = [neighbor for neighbor in graph[node_key]
                        if graph.nodes[neighbor]['element'] not in 'CH']
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
                       graph.nodes[node_key]['element'] not in 'CH',
                       cycle_invar))
    return invariant


class RounderMixIn:
    def predict(self, *args, **kwargs):
            predicted = super().predict(*args, **kwargs)
            return self.do_round(predicted)

    @staticmethod
    def do_round(y):
        return y.round().astype(int)


def first_letter(string):
    for char in string:
        if char.isalpha():
            return char


def add_element(mol):
    for idx in mol:
        node = mol.nodes[idx]
        if 'element' not in node:
            node['element'] = first_letter(node['name'])

#
#def draw_mol(mol):
#    labels = nx.get_node_attributes(mol, 'name')
#    nx.draw_networkx(mol, labels=labels)


def featurize_all(molecules, num_features, fingerprint_size, condense=True):
    fingerprinter = XCFPFingerprinter(fingerprint_size, invariant=invariant)
    features = []
    results = []
    
    for name, aa_mol, cg_mol, table, mapdict in molecules:
        add_element(aa_mol)
        fingerprinter.fingerprint(aa_mol)
        mol = aa_mol.copy()
        
        for node_idx in mol:
            col_idx = table.atomnums.index(node_idx)
            vals = table.values[:, col_idx]
            mol.nodes[node_idx]['weights'] = vals
    
        for at1, at2 in mol.edges:
            w1 = mol.nodes[at1]['weights']
            w2 = mol.nodes[at2]['weights']
            w = np.abs(w2 - w1).sum()
            mol.edges[at1, at2]['weight'] = w/2  # /2, because where one bead ends the next one begins
        for node1, node2, cut in mol.edges(data='weight'):
            feature1 = fingerprinter._per_node[node1]
            feature2 = fingerprinter._per_node[node2]
            feat_arr = np.zeros(num_features*2, dtype=int)
            for feat in feature1:
                feat_arr[feat % num_features] += 1
            for feat in feature2:
                feat_arr[feat % num_features + num_features] += 1
            features.append(feat_arr)
            results.append(cut)
    #    print('{: >25} {:>2} {:5.2f} {:5.2f}'.format(str(name), len(cg_mol), sum(nx.get_node_attributes(mol, 'ncuts').values()), sum(nx.get_edge_attributes(mol, 'weight').values())))

    results = np.array(results)
    features = np.array(features, dtype=float)
    if condense:
        mean_features = np.unique(features, axis=0)
        mean_results = np.empty((mean_features.shape[0]))
        weights = np.empty_like(mean_results)
        
        for idx, f in enumerate(mean_features):
            res = results[np.all(features == f, axis=1)]
            mean_results[idx] = res.mean()
#            std = res.std()
#            if len(res) == 1:
#                weights[idx] = 1
#            elif abs(std) < 1e-7:
#                weights[idx] = len(res) * 1e2  # std = 1e-1
#            else:
#                weights[idx] = len(res) / std**2
            weights[idx] = len(res)
#            print('{:>4} {:.2f} {:.2f}'.format(len(res), res.mean(), weights[idx]))
        return mean_features, mean_results, weights
    else:
        return features, results, np.ones_like(results)


#scaler = preprocessing.RobustScaler()
# Hyperparameters are determined by hyperopt_weights_edge.py
#MODEL = pipeline.Pipeline(steps=[('scale', scaler),
#                                 ('estimator', svm.SVR())])
MODEL = ensemble.RandomForestRegressor(n_estimators=500)
MODEL.set_params(**dict((('criterion', 'mse'),
   ('max_features', 1),
   ('max_depth', 48),
   ('min_samples_split', 2),
   ('min_samples_leaf', 1),
   ('bootstrap', True))))
NUM_FEATURES = 7
FINGERPRINT_SIZE = 3
CONDENSE = True
BASE_PATH = '/home/.../Documents/database'
XLS_FILE = os.path.join(BASE_PATH, 'DRUGS-06.xlsx')
AA_DIR = os.path.join(BASE_PATH, 'atomistic')
CG_DIR = os.path.join(BASE_PATH, 'Martini')
MAP_DIR = os.path.join(BASE_PATH, 'mapping')

if __name__ == '__main__':

    # If you change OUTPUT_FILENAME here, you must also change
    # it in predict_num_beads.py
    OUTPUT_FILENAME = 'weight_edge_predictor.gz'

    molecules = parse_db(XLS_FILE, AA_DIR, CG_DIR, MAP_DIR)

    features, results, weights = featurize_all(molecules, NUM_FEATURES, FINGERPRINT_SIZE, CONDENSE)

    trainX, testX, trainY, testY = model_selection.train_test_split(features,
                                                                    results,
                                                                    test_size=0.2)
    train = trainX, trainY
    test = testX, testY

    cv_out = model_selection.cross_validate(MODEL, *train, cv=4)
    print('Cross validation scores on the training set:')
    print(cv_out['test_score'])

    MODEL.fit(*train)

    joblib.dump(MODEL, OUTPUT_FILENAME, compress=True)

    print('Score for the validation set:')
    print(MODEL.score(*test))

    plt.scatter(train[1], MODEL.predict(train[0]), c='b', label='train')
    plt.scatter(test[1], MODEL.predict(test[0]), c='r', label='test')
    xmin, xmax = plt.xlim()

    plt.plot([xmin, xmax], [xmin, xmax], '--')
    plt.xlim(xmin, xmax)
    plt.legend()
    plt.show()
