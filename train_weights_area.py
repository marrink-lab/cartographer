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
from sklearn import model_selection, ensemble, svm, preprocessing, pipeline
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
                       graph.nodes[node_key]['element'] not in 'CH',
                       graph.nodes[node_key]['charge'],
                       cycle_invar))
    return invariant


#
#def draw_mol(mol):
#    labels = nx.get_node_attributes(mol, 'name')
#    nx.draw_networkx(mol, labels=labels)


def featurize_all(molecules, num_features, neighborhood_size, fingerprint_size, condense=True):
    fingerprinter = XCFPFingerprinter(fingerprint_size, invariant=invariant)
    features = []
    results = []
    
    for name, aa_mol, cg_mol, table, mapdict in molecules:
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
        for node, feats in fingerprinter._per_node.items():
            feat_arr = np.zeros(num_features, dtype=int)
            for feat in feats:
                feat_arr[feat % len(feat_arr)] += 1
            neighborhood = nx.ego_graph(mol, node, radius=neighborhood_size)
            weights = nx.get_edge_attributes(neighborhood, 'weight')
            num_cuts = sum(weights.values())
            mol.nodes[node]['ncuts'] = num_cuts
            features.append(feat_arr)
            results.append(num_cuts)
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
            std = res.std()
            if len(res) == 1:
                weights[idx] = 1
            elif abs(std) < 1e-7:
                weights[idx] = len(res) * 1e2  # std = 1e-1
            else:
                weights[idx] = len(res) / std**2
#            weights[idx] = len(res)
#            print('{:>4} {:.2f} {:.2f}'.format(len(res), res.mean(), weights[idx]))
        return mean_features, mean_results, weights
    else:
        return features, results, np.ones_like(results)


#scaler = preprocessing.RobustScaler()
# Hyperparameters are determined by hyperopt_.py
#MODEL = pipeline.Pipeline(steps=[('scale', scaler),
#                                 ('estimator', svm.SVR())])
MODEL = ensemble.RandomForestRegressor(n_estimators=500)
#
#MODEL.set_params(**dict((('estimator__epsilon', 0.5623413251903491),
#   ('estimator__kernel', 'rbf'),
#   ('estimator__C', 1.0),
#   ('estimator__shrinking', False),
#   ('estimator__gamma', 1.0))))

NUM_FEATURES = 7
NEIGHBORHOOD_SIZE = 2
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
    OUTPUT_FILENAME = 'weight_area_predictor.gz'

    molecules = parse_db(XLS_FILE, AA_DIR, CG_DIR, MAP_DIR)

    features, results, weights = featurize_all(molecules, NUM_FEATURES, NEIGHBORHOOD_SIZE, FINGERPRINT_SIZE, CONDENSE)

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
