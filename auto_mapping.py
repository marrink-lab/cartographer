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

from train_num_beads import FingerPrinter, SVRR
from predict_num_beads import predict_mol as num_beads
from predict_weight_edge_class import predict_mol as edge_weights
from pysmiles import read_smiles

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

from collections import defaultdict


def draw_molecule(mol, clusters, edge_widths=None):
    if edge_widths is None:
        widths = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    else:
        widths = edge_widths
    elems = nx.get_node_attributes(mol, 'element')

    labels = {}
    for n_idx, elem in elems.items():
        labels[n_idx] = '{}{}'.format(elem, n_idx)

    edges = defaultdict(list)
    for idx, jdx, order in mol.edges(data='order'):
        edges[widths[order]].append((idx, jdx))

    pos = nx.spring_layout(mol, weight='None', iterations=1000)

    nx.draw_networkx_nodes(mol, pos=pos, cmap='tab20c',
                           node_color=[clusters[idx] for idx in mol])
    nx.draw_networkx_labels(mol, pos=pos, labels=labels)
    for width, edgelist in edges.items():
        nx.draw_networkx_edges(mol, pos, edgelist=edgelist, width=width)


def map_molecule(mol, nbeads=None, edges=None, delta=1):
    if nbeads is None:
        nbeads = num_beads(mol)[0] + 1
    if edges is None:
        edges = edge_weights(mol)

    for edge, weight in edges.items():
        if weight == 0:
            weight = 0.01
#        elif weight == 1:
#            weight = 1.1
#        edges[edge] = weight + 0.1

    nx.set_edge_attributes(mol, edges, name='weight')

    ordering = list(mol.nodes)

    distance_mat = nx.to_scipy_sparse_matrix(mol, nodelist=ordering, weight='weight')
    similarity_matrix = distance_mat
    idxs = distance_mat.nonzero()
    similarity_matrix[idxs] = 1/distance_mat[idxs]
#    similarity_matrix[idxs] = np.exp(-distance_mat[idxs].A**2/(2*delta**2)) * 100
#    print(similarity_matrix)

    sgp = SpectralClustering(n_clusters=nbeads, affinity='precomputed',
                             assign_labels='discretize', n_init=250)
    clusters = sgp.fit_predict(similarity_matrix)
    clusters = dict(zip(ordering, clusters))
    return clusters


mol = read_smiles('CC1=C(C2=CC3=C(C(=C([N-]3)C=C4C(=C(C(=N4)C=C5C(=C(C(=N5)C=C1[N-]2)C=C)C)C=C)C)C)CCC(=O)O)CCC(=O)O', explicit_H=False)

clusters = map_molecule(mol, delta=0.1)
print(max(clusters.values()) + 1)
draw_molecule(mol, clusters)
plt.show()
