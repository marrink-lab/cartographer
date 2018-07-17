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

from pprint import pprint

from evolutionary_search import EvolutionaryAlgorithmSearchCV
import numpy as np
from sklearn import model_selection, utils

from parse_db import parse_db
from train_weights_area import (MODEL, featurize_all, NUM_FEATURES, XLS_FILE,
                                AA_DIR, CG_DIR, MAP_DIR, NEIGHBORHOOD_SIZE,
                                FINGERPRINT_SIZE, CONDENSE)


molecules = parse_db(XLS_FILE, AA_DIR, CG_DIR, MAP_DIR)

features, results, names = featurize_all(molecules, NUM_FEATURES, NEIGHBORHOOD_SIZE, FINGERPRINT_SIZE, CONDENSE)
trainX, testX, trainY, testY = model_selection.train_test_split(features,
                                                                results,
                                                                test_size=0.2)
train = trainX, trainY
test = testX, testY


param_grid = [
              {                                                              
              #'n_estimators': np.logspace(1, 3, num=15, base=10, dtype=int),   
              'criterion': ['mae', 'mse'],                                      
              'max_features': list(range(1, features.shape[1]+1)),              
              'max_depth': np.arange(1, 101, dtype=int),                        
              'min_samples_split': np.arange(2, 11, dtype=int),                 
              'min_samples_leaf': np.arange(1, 11, dtype=int),                  
              'bootstrap': [True, False]                                        
              }
             ]

outputs = []

ncpus = utils.cpu_count()
for grid in param_grid:
    cv = EvolutionaryAlgorithmSearchCV(MODEL, grid, scoring='r2', cv=4,
                                       verbose=1,
                                       population_size=(75//ncpus) * ncpus,
                                       gene_mutation_prob=0.10,
                                       gene_crossover_prob=0.5,
                                       tournament_size=3,
                                       generations_number=25,
                                       n_jobs=ncpus,
                                       error_score=-10)
    cv.fit(*train)
    MODEL.set_params(**cv.best_params_)
    MODEL.fit(*train)
    test_score = MODEL.score(*test)
    outputs.append((test_score, tuple(cv.best_params_.items())))
outputs.sort()
pprint(outputs)
