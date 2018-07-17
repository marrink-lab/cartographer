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

import networkx as nx
from collections import defaultdict, namedtuple


class XCFPFingerprinter:
    """
    A class for generating extended connectivty fingerprints (`ECFPs`_) from
    graphs. You can change the hashing algorithm used by setting the class
    attribute :attr:`hashf`. By default it uses the function :func:`hash`.

    Attributes
    ----------
    hashf : :func:`hash`
        The function used for hashing
    identifier : :obj:`namedtuple`
        A namedtuple with fields ``value`` and ``edges``, containing the hash
        value and the corresponding edges respectively.
    radius : :obj:`int`
        See ``Parameters``.
    invariant : :obj:`callable`
        See ``Parameters``.
    bond_order : :obj:`callable`
        See ``Parameters``.

    Parameters
    ----------
    radius : :obj:`int`
        The number of iterations to perform. Note that this is a radius and
        not a diameter, so ``radius=3`` corresponds with ECFP_6.
    invariant : ``f(graph, node_key) -> hashable``, optional
        A callable that takes a graph and a node key for which to calculate the
        node invariants. Returns a hashable object usually a tuple of relevant
        node properties such as the number of neighbours and the element.
        Defaults to :meth:`default_invariant`.
    bond_order : ``f(graph, node_key1, node_key2) -> hashable``, optional
        A callable that takes a graph and two node keys denoting an edge in
        the graph. Should return a hashable denoting the type of edge; usually
        an :obj:`int`. Defaults to :meth:`default_bond_order`.

    .. _ECFPs:
        https://pubs.acs.org/doi/full/10.1021/ci100050t
    """
    hashf = hash
    identifier = namedtuple('Identifier', ['value', 'edges'])

    def __init__(self, radius, invariant=None, bond_order=None):
        if invariant is None:
            invariant = self.default_invariant
        if bond_order is None:
            bond_order = self.default_bond_order
        self.invariant = invariant
        self.bond_order = bond_order
        self.radius = radius

    def fingerprint(self, graph):
        """
        Calculates the fingerprint for `graph`.

        Parameters
        ----------
        graph : :class:`~nx.Graph`
            The graph for which to calculate the fingerprint.

        Returns
        -------
        dict of int: int
            The resulting fingerprint. Keys are the feature hashes, values how
            often they were found.
        """
        # These 2 are persistent for this molecule
        self._fingerprint = defaultdict(int)
        self._known_features = dict()
        self._per_node = defaultdict(list)

        # These 2 are recalculated every iteration
        # The identifiers for the current iteration per node
        self._identifiers = {}
        # The edges described by these identifiers.
        self._covered_edges = defaultdict(set)

        # Initial assignment stage
        for node_key in graph:
            hash_val = self.hashf(self.invariant(graph, node_key))
            self._identifiers[node_key] = self.identifier(hash_val, set())
        self._store()
        # Iterative updating stage
        for iternum in range(self.radius):
            self._do_iter(graph, iternum)
            self._store()
        # Duplicate removal is dealt with by _store.
        return dict(self._fingerprint)

    def _do_iter(self, graph, iternum=0):
        """
        Does a single iteration of the xCFP algorithm. Results are stored in
        :attr:`_identifiers` and :attr:`_covered_edges`.

        Parameters
        ----------
        graph : :class:`~nx.Graph`
            The graph to work on.
        iternum : :obj:`int`
            The current iteration.
        """
        # We need to store these separately for a while because we need to
        # update all nodes simultaneously.
        new_identifiers = {}
        new_covered_edges = {}
        for node_key in graph:
            ids = [iternum, self._identifiers[node_key].value]
            order = self.sort_neighbours(graph, node_key)
            edges = self._covered_edges[node_key].copy()
            for neighbour in order:
                ids.append(self.bond_order(graph, node_key, neighbour))
                ids.append(self._identifiers[neighbour].value)
                # Remember edges to my nieghbours...
                edges.add(frozenset((node_key, neighbour)))
                # and all the edges described by my neighbour's hash value.
                edges.update(self._covered_edges[neighbour])
            hash_val = self.hashf(tuple(ids))
            new_identifiers[node_key] = self.identifier(hash_val, edges)
            new_covered_edges[node_key] = edges
        self._identifiers = new_identifiers
        self._covered_edges = new_covered_edges

    def _store(self):
        """
        Store the identifiers from this round in :attr:`_fingerprint`, after
        dealing with structural duplication.
        """
#        for node_key, identifier in self._identifiers.items():
#            self._per_node[node_key].append(identifier.value)
        for identifier in sorted(self._identifiers.items(), key=lambda i: i[1].value):
            node_key, identifier = identifier
            feature = frozenset(identifier.edges)
            # feature is an empty set at iteration 0.
            if not feature or feature not in self._known_features:
                self._fingerprint[identifier.value] += 1
                self._known_features[feature] = identifier.value
            self._per_node[node_key].append(self._known_features.get(feature, identifier.value))

    def sort_neighbours(self, graph, root_node):
        """
        Sorts neighbours of `root_node` based on `current_identifiers`
        according to the xCFP algorithm. Uses :meth:`bond_order` to sort by
        bonds first, and :attr:`_identifiers` to sort by current feature values
        next.

        Parameters
        ----------
        graph : :class:`nx.Graph`
            Graph to work on.
        root_node
            Node key of the node whose neighbours we're sorting

        Returns
        -------
        list
            Sorted list of `root_node`'s neighbours.
        """
        def keyfunc(node_key):
            return (self.bond_order(graph, root_node, node_key),
                    self._identifiers[node_key].value)
        return sorted(graph[root_node], key=keyfunc)

    @staticmethod
    def default_invariant(graph, node_key):
        """
        Calculates the invariant for the node identified by `node_key` in
        `graph`. The invariant is a tuple of:
            - the number of neighbours;
            - the element;
            - the charge;
            - 1 if the node is in at least one ring, 0 otherwise.

        For the ECFP algorithm it should be:

            - the number of immediate neighbors who are “heavy” (non-hydrogen)
              atoms;
            - the valence minus the number of hydrogens;
            - the atomic number;
            - the atomic mass;
            - the atomic charge;
            - the number of attached hydrogens (both implicit and explicit);
            - 1 if the node is in at least one ring, 0 otherwise.

        Attributes
        ----------
        graph : :class:`nx.Graph`
            Graph to work with
        node_key
            The node for which to calculate the invariant.

        Returns
        -------
        tuple
            The number of neighbours, the element, the charge, and 1 iff the
            node is in a cycle; 0 otherwise.
        """
        cycles = nx.cycle_basis(graph)
        invariant = tuple((len(graph[node_key]),  # number of neighbours
                           graph.nodes[node_key]['element'],
                           graph.nodes[node_key].get('charge', 0),
                           1 if any(node_key in cycle for cycle in cycles) else 0))
        return invariant

    @staticmethod
    def default_bond_order(graph, node_key1, node_key2):
        """
        Returns an integer identifying the bond order between `node_key1` and
        `node_key2` in `graph`. Currently always returns 1.
        Should be 1, 2, 3, 4 for single, double, triple and aromatic
        respectively for the ECFP algorithm.
        """
        order = graph.edges[node_key1, node_key2].get('order', 1)
        if order == 1.5:
            order = 4
        return order


if __name__ == '__main__':
    mol = nx.Graph()
    mol.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)])
    for k in mol:
        mol.nodes[k]['element'] = 'C'
        mol.nodes[k]['charge'] = 0

    mol.nodes[5]['element'] = 'O'
    mol.nodes[4]['element'] = 'N'
    ecfp0 = XCFPFingerprinter(0)
    print(ecfp0.fingerprint(mol))
    print(ecfp0._per_node)
    print()

    ecfp2 = XCFPFingerprinter(1)
    print(ecfp2.fingerprint(mol))
    print(ecfp2._per_node)
    print()

    ecfp4 = XCFPFingerprinter(2)
    print(ecfp4.fingerprint(mol))
    print(ecfp4._per_node)
    print()

    ecfp6 = XCFPFingerprinter(3)
    fingerprint = ecfp6.fingerprint(mol)
    print(fingerprint)
    print(ecfp6._per_node)
    for fps in ecfp6._per_node.values():
        for fp in fps:
            if fp not in fingerprint:
                print('Uhow! {}'.format(fp))
