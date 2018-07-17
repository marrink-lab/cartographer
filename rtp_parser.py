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

from collections import defaultdict

import networkx as nx
# RTP parser

def parse_rtp(filename, remove_H=True):
    # Everything else is a moleculename
    valid_contexts = 'bondedtypes atoms bonds exclusions angles dihedrals impropers'.split()
    molecules = defaultdict(nx.Graph)
    with open(filename) as file:
        context = None
        molname = None
        name_to_num = {}
        for line in file:
            line, *comment = line.split(';', 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line.strip('[ ]')
                if section in valid_contexts:
                    context = section
                else:
#                    molecules[molname] = nx.convert_node_labels_to_integers(molecules[molname], label_attribute='name')
                    molname = section
                    idx = 1
            elif line.startswith('#'):
#                print(section, line)
                continue
            elif context == 'atoms':
                name, type_, charge, chgrp = line.split()
                if remove_H and name.startswith('H'):
                    continue
                molecules[molname].add_node(idx, name=name)
                name_to_num[name] = idx
                idx += 1
            elif context == 'bonds' or context == 'constraints':
                at1, at2, *params = line.split()
                if '+' in at1 or '-' in at1 or '+' in at2 or '-' in at2:
                    continue
                if remove_H and (at1.startswith('H') or at2.startswith('H')):
                    continue
                molecules[molname].add_edge(name_to_num[at1], name_to_num[at2])
    molecules = dict(molecules)
    return molecules

def parse_itp(filename, remove_H=True):
    molecules = defaultdict(nx.Graph)
    with open(filename) as file:
        context = None
        molname = None
        for line in file:
            line, *comment = line.split(';', 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                section = line.strip('[ ]')
                context = section
            elif line.startswith('#'):
#                print(section, line)
                continue
            elif context == 'moleculetype':
                molname, *_ = line.split()
            elif context == 'atoms':
                idx, type_, residx, resname, name, chgrp, *charge = line.split()
                attrs = dict(name=name)
                idx = int(idx)
                if idx not in molecules[molname]:
                    molecules[molname].add_node(idx, **attrs)
                else:
                    raise KeyError(filename)
                    molecules[molname].nodes[idx].update(attrs)
            elif context == 'bonds' or context == 'constraints':
                at1, at2, *params = line.split()
                at1, at2 = int(at1), int(at2)
                molecules[molname].add_edge(at1, at2)
    for molname, mol in molecules.items():
        to_remove = set()
        for node_idx in mol:
            name = mol.nodes[node_idx]['name']
            if remove_H and name.startswith('H'):
                to_remove.add(node_idx)
        mol.remove_nodes_from(to_remove)
#        idx_to_name = nx.get_node_attributes(mol, 'name')
#        molecules[molname] = nx.relabel_nodes(mol, idx_to_name)
    return dict(molecules)
