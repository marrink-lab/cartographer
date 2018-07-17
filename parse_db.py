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

from rtp_parser import parse_itp, parse_rtp
from mapping_parser import mapping_parser

from collections import namedtuple
from functools import lru_cache
import os.path as osp

import networkx as nx
import openpyxl
from pysmiles import read_smiles


COLUMNS = namedtuple('Row', 'name skip aa_name cg_name aa_file cg_file '
                            'mapping_file smiles inchi inchikey comments '
                            'references'.split())
PARSERS = {'itp': parse_itp, 'rtp': parse_rtp}


def use_row(row):
    return not row.skip


def get_mol(filename, name, skip_H=True):
    return parse_file(filename, skip_H)[name]


@lru_cache()
def parse_file(filename, skip_H=True):
    base, ext = osp.splitext(filename)
    ext = ext.lstrip('.')
    mols = PARSERS[ext](filename, skip_H)
    return mols


def get_mapping(filename, normalize=False):
    names, table, mapdict = mapping_parser(filename, normalize)
    return names, table, mapdict


def first_letter(string):
    if string.upper().startswith('CL'):
        return 'Cl'
    elif string.upper().startswith('FE'):
        return 'Fe'
    elif string.upper().startswith('MG'):
        return 'Mg'
    for char in string:
        if char.isalpha():
            return char


def add_element(mol):
    for idx in mol:
        node = mol.nodes[idx]
        if 'element' not in node:
            node['element'] = first_letter(node['name'])


def parse_db(xls_file, aa_dir, cg_dir, map_dir):
    wb = openpyxl.load_workbook(xls_file, read_only=True)
    ws = wb.active
    molecules = []
    for row in ws.iter_rows(min_row=2):
        try:
            row_vals = [cell.value for cell in row]
            if len(row_vals) > len(COLUMNS._fields):
                row_vals = row_vals[:len(COLUMNS._fields)]
            else:
                row_vals += [None] * (len(COLUMNS._fields) - len(row_vals))
            row = COLUMNS(*row_vals)
        except Exception as err:
            print([cell.value for cell in row])
            raise
        if use_row(row):
            aa_mol = get_mol(osp.join(aa_dir, row.aa_file), row.aa_name)
            cg_mol = get_mol(osp.join(cg_dir, row.cg_file), row.cg_name)
            map_names, table, mapdict = get_mapping(osp.join(map_dir, row.mapping_file), True)
            add_element(aa_mol)
            if set((row.aa_name, row.cg_name)) < set(map_names):
                print('Mapping names seem off:')
                print(map_names, row.aa_name, row.cg_name)
            if row.smiles:
                smiles_mol = read_smiles(row.smiles, explicit_hydrogen=False, zero_order_bonds=False)
                for node_idx in smiles_mol:
                    smiles_mol.nodes[node_idx]['element'] = smiles_mol.nodes[node_idx]['element'].capitalize()
                matcher = nx.isomorphism.GraphMatcher(aa_mol, smiles_mol, node_match=nx.isomorphism.categorical_node_match('element', 'H'))
                match = list(matcher.match())
                if not match:
                    print('Smiles and ITP for {} do not match'.format(row.aa_name))
                    print(len(aa_mol), aa_mol.nodes(data='element'))
                    print(len(aa_mol.edges), aa_mol.edges)
                    print(len(smiles_mol), smiles_mol.nodes(data='element'))
                    print(len(smiles_mol.edges), smiles_mol.edges)
                else:
                    match = match[0]
                    for aa_idx, smi_idx in match.items():
                        smi_node = smiles_mol.nodes[smi_idx]
                        aa_node = aa_mol.nodes[aa_idx]
                        for attr, val in smi_node.items():
                            aa_node[attr] = val
                    for aa_idx, aa_jdx in aa_mol.edges:
                        smi_idx, smi_jdx = match[aa_idx], match[aa_jdx]
                        for attr, val in smiles_mol.edges[smi_idx, smi_jdx].items():
                            aa_mol.edges[aa_idx, aa_jdx][attr] = val
                    aa_mol.graph['smiles'] = row.smiles
            else:
                print('smiles for mol {} is missing.'.format(row.aa_name))
            molecules.append(((row.aa_name, row.cg_name), aa_mol, cg_mol, table, mapdict))
    print('Read {} molecules'.format(len(molecules)))
    return molecules
