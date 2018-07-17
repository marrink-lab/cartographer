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

from collections import defaultdict, namedtuple
import numpy as np
# Backmapping files parser


def mapping_parser(filename, normalize=False):
    """
    Parses the `atoms` section in the backmapping file `filename`. Also uses
    information from the `martini` section.

    Parameters
    ----------
    filename : str
        The file to parse
    normalize : bool
        Whether or not to normalize how often an atom can contribute to a bead.
        If normalized, atoms shared between beads will have a fractional
        contribution to their beads; summing to 1.

    Returns
    -------
    (np.array[num_beads, num_atoms], dict{bead_name: [atomnames]})
        The dtype of the numpy array depends on the normalize argument. `int`
        if `False`, `float` otherwise.
    """
    context = None

    mapping = defaultdict(list)
    beads = []
    atoms = []
    nums = []
    name = None
    with open(filename) as file:
        for line in file:
            if ';' in line:
                line, comment = line.split(';', 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith(']'):
                context = line.strip('[ ]')
                continue
            if context == 'martini':
                beads.extend(line.split())
            elif context == 'molecule':
                names = line.split()
            elif context == 'atoms':
                num, atom, *martini = line.split()
#                if atom.startswith('H'):
#                    continue
                atoms.append(atom)
                nums.append(int(num))
                for bead in martini:
                    mapping[bead].append(int(num))
    nums, idxs = np.unique(nums, return_index=True)
    atoms = np.array(atoms)[idxs]
    
    mapping = dict(mapping)

    if set(mapping.keys()) > set(beads):
#    if not len(mapping) == len(beads):
        print(filename)
        print(beads)
        print(mapping.keys())
        raise AssertionError
    Table = namedtuple('Table', ['beadnames', 'atomnames', 'atomnums', 'values'])
    beads = np.array(beads)
    output = np.zeros((len(beads), len(nums)), dtype=int)
    for beadname, atomnums in mapping.items():
#        print(beadname, atomnums)
        bead_idx = np.where(beads == beadname)
        for atomnum in atomnums:
            atom_idx = np.where(nums == atomnum)
#            print(bead_idx, atom_idx)
            output[bead_idx, atom_idx] += 1

    if normalize:
        output = output.astype(float)
        norm = output.sum(axis=0)
        mask = norm != 0
        # Alternatively, divide by 0, and do output[~np.isfinite(output)] = 0
        output[:, mask] /= norm[np.newaxis, mask]
    output = Table(atomnames=list(atoms), beadnames=list(beads), atomnums=list(nums), values=output)
    return names, output, mapping
