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
from sklearn import model_selection, ensemble, pipeline, svm, preprocessing
from sklearn.externals import joblib
from discretization import KBinsDiscretizer

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


def first_letter(string):
    for char in string:
        if char.isalpha():
            return char


def add_element(mol):
    for idx in mol:
        node = mol.nodes[idx]
        if 'element' not in node:
            node['element'] = first_letter(node['name'])


def draw_mol(mol):
    labels = nx.get_node_attributes(mol, 'name')
    nx.draw_networkx(mol, labels=labels)


def featurize(mol, feature_size=7, fingerprint_size=2):
    fingerprinter = XCFPFingerprinter(fingerprint_size, invariant=invariant)
    fingerprinter.fingerprint(mol)

    feature_per_edge = {}
    for (u, v) in mol.edges:
        feat_arr = np.zeros(feature_size*2, dtype=int)
        feature1 = fingerprinter._per_node[u]
        feature2 = fingerprinter._per_node[v]
        for feat in feature1:
            feat_arr[feat % feature_size] += 1
        for feat in feature2:
            feat_arr[feat % feature_size + feature_size] += 1
        feature_per_edge[(u, v)] = feat_arr
    return feature_per_edge


def remove_duplicates(features, results, avg_func=np.mean, weight_func=len):
    mean_features = np.unique(features, axis=0)
    mean_results = np.empty((mean_features.shape[0]))
    weights = np.empty_like(mean_results)

    for idx, f in enumerate(mean_features):
        res = results[np.all(features == f, axis=1)]
        mean_results[idx] = avg_func(res)
        weights[idx] = weight_func(res)
    return mean_features, mean_results, weights


def embed_mapping(mol, table, attrname='weight'):
    for node_idx in mol:
        col_idx = table.atomnums.index(node_idx)
        vals = table.values[:, col_idx]
        mol.nodes[node_idx][attrname] = vals
    for at1, at2 in mol.edges:
        w1 = mol.nodes[at1][attrname]
        w2 = mol.nodes[at2][attrname]
        w = np.abs(w2 - w1).sum()
        # /2, because where one bead ends the next one begins
        cut = w/2
        mol.edges[at1, at2][attrname] = cut


def featurize_all(molecules, num_features, fingerprint_size, condense=True):
    features = []
    results = []

    for name, aa_mol, cg_mol, table, mapdict in molecules:
        add_element(aa_mol)
        mol = aa_mol.copy()
        embed_mapping(mol, table, 'weight')

        feature_per_edge = featurize(mol, num_features, fingerprint_size)
        for (at1, at2), feat_arr in feature_per_edge.items():
            features.append(feat_arr)
            results.append(mol[at1][at2]['weight'])

    results = np.array(results)
    features = np.array(features)

    if condense:
        features, results, weights = remove_duplicates(features, results)
    else:
        weights = np.ones_like(results)
    return features, results, weights


NUM_FEATURES = 7
FINGERPRINT_SIZE = 3
NBINS = 3
CONDENSE = True
FILENAME = 'weight_edge_predictor.gz'

BASE_PATH = '/home/.../Documents/database'
XLS_FILE = os.path.join(BASE_PATH, 'DRUGS-06.xlsx')
AA_DIR = os.path.join(BASE_PATH, 'atomistic')
CG_DIR = os.path.join(BASE_PATH, 'Martini')
MAP_DIR = os.path.join(BASE_PATH, 'mapping')

# Hyperparameters are determined by hyperopt_weights_edge.py
DISCRETIZER = KBinsDiscretizer(n_bins=NBINS, encode='ordinal')
# Note that we can't make a fingerprinter pipeline like we did with num_beads,
# since there'll be multiple features per molecule, and we need to remove
# duplicates. And the problem there is that you can't change the number of
# samples; partly because this would mess with e.g. cross validation and
# train/test splitting.
#rfr = ensemble.RandomForestClassifier(n_estimators=500)
#MODEL = rfr
#MODEL.set_params(**dict((('criterion', 'gini'),
#   ('max_features', 3),
#   ('max_depth', 22),
#   ('min_samples_split', 6),
#   ('min_samples_leaf', 1),
#   ('bootstrap', False))))

scaler = preprocessing.RobustScaler()
svc = svm.SVC()
MODEL = pipeline.Pipeline(steps=[('scale', scaler),
                                 ('estimator', svc)])


extension = 1/(NBINS-1)
# Don't fit on the actual data to make sure it's transferable to the
# predictor. In addition, a bond is cut or not, so anything over 1 is silly
# anyway. Extend the range a little bit over (0-1) so that the outermost bins
# are centered at 0 and 1 respectively.
DISCRETIZER.fit(np.reshape((-extension/2, 1 + extension/2), (-1, 1)))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    molecules = parse_db(XLS_FILE, AA_DIR, CG_DIR, MAP_DIR)

    features, results, weights = featurize_all(molecules, NUM_FEATURES,
                                               FINGERPRINT_SIZE, CONDENSE)

    results = DISCRETIZER.transform(results.reshape((-1, 1))).ravel()

    split = model_selection.train_test_split(features, results, weights,
                                             test_size=0.2)
    trainX, testX, trainY, testY, trainW, testW = split
    train = trainX, trainY#, trainW
    test = testX, testY#, testW

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
