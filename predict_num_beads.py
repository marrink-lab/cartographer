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

from sklearn.externals import joblib

from pysmiles import read_smiles
# SVRR is the model used, and it's definition is needed to load the pickled
# file
from train_num_beads import FingerPrinter, SVRR
from train_num_beads import FILENAME, featurize

clf = joblib.load(FILENAME)
def predict_mol(mol):
    result = clf.predict(mol)
    return result


if __name__ == '__main__':

    SMILES = ['CCN1CCC(=CC1)c1cccc(c1)OC',
              'OC(=O)CN(S(=O)(=O)c1ccccc1N)c1ccccc1',
              'FC(OC([C@@H](Cl)F)(F)F)F',
              'Cc1cn([C@@H]2C=C[C@@H](O2)CO[P@](=O)(C(F)F)O)c(=O)[nH]c1=O',
              'CCOC(=O)c1ccc(cc1)[C@@H]1OC[C@@H]2[C@@H](O1)[C@H](O)[C@H]([C@H](O2)OC)O',
              'COC(=O)Nc1ccc(c(c1)C1=[N]=[C]2=C(C)C=CC=C2S1)Cl']

    for smile in SMILES:
        mol = read_smiles(smile, explicit_H=False)

        num_beads = predict_mol(mol)
        print(num_beads)
