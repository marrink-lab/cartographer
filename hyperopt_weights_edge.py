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
from sklearn import model_selection, utils, base

from parse_db import parse_db
from train_weights_edge_class import (MODEL, featurize_all, NUM_FEATURES, XLS_FILE,
                                AA_DIR, CG_DIR, MAP_DIR, FINGERPRINT_SIZE,
                                CONDENSE, DISCRETIZER)


molecules = parse_db(XLS_FILE, AA_DIR, CG_DIR, MAP_DIR)

features, results, names = featurize_all(molecules, NUM_FEATURES, FINGERPRINT_SIZE, CONDENSE)
results = DISCRETIZER.transform(results.reshape((-1, 1))).ravel()

trainX, testX, trainY, testY = model_selection.train_test_split(features,
                                                                results,
                                                                test_size=0.2)
train = trainX, trainY
test = testX, testY

#param_grid = [
#              {                                                              
#              #'n_estimators': np.logspace(1, 3, num=15, base=10, dtype=int),   
#              'criterion': ['gini', 'entropy'],                                      
#              'max_features': list(range(1, features.shape[1]+1)),              
#              'max_depth': np.arange(1, 101, dtype=int),                        
#              'min_samples_split': np.arange(2, 11, dtype=int),                 
#              'min_samples_leaf': np.arange(1, 11, dtype=int),                  
#              'bootstrap': [True, False]                                        
#              }
#             ]
param_grid = [
              {
               'estimator__C': np.logspace(-5, 5, num=25, base=10),
               'estimator__kernel': ['rbf'],
               'estimator__decision_function_shape': ['ovr', 'ovo'],
               'estimator__gamma': np.logspace(-5, 5, num=25, base=10),
               'estimator__shrinking': [True, False],
              },
              {
               'estimator__C': np.logspace(-5, 5, num=25, base=10),
               'estimator__kernel': ['poly'],
               'estimator__decision_function_shape': ['ovr', 'ovo'],
               'estimator__degree': [2, 3, 4],
               'estimator__gamma': np.logspace(-5, 5, num=15, base=10),
               'estimator__coef0': np.logspace(-3, 3, num=15, base=10),
               'estimator__shrinking': [True, False],
              },
              {
               'estimator__C': np.logspace(-5, 5, num=25, base=10),
               'estimator__kernel': ['sigmoid'],
               'estimator__decision_function_shape': ['ovr', 'ovo'],
               'estimator__gamma': np.logspace(-5, 5, num=15, base=10),
               'estimator__coef0': list(-np.logspace(-5, 5, num=15, base=10)) + [0] + list(np.logspace(-3, 3, num=15, base=10)),
               'estimator__shrinking': [True, False],
              },
              {
               'estimator__C': np.logspace(-5, 5, num=25, base=10),
               'estimator__kernel': ['linear'],
               'estimator__decision_function_shape': ['ovr', 'ovo'],
               'estimator__shrinking': [True, False],
              },
             ]
# Non rbf kernels have convergence issues with some parameters. So set a max_iter
# to something reasonable to make sure it stops if it's wandering too far off.
MODEL.set_params(estimator__max_iter=int(1e7), estimator__cache_size=1000)

# ... and squelsh the warnings.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=Warning)

outputs = []

ncpus = utils.cpu_count()
for grid in param_grid:
    cv = EvolutionaryAlgorithmSearchCV(MODEL, grid, cv=4,
                                       verbose=1,
                                       population_size=(75//ncpus) * ncpus,
                                       gene_mutation_prob=0.10,
                                       gene_crossover_prob=0.5,
                                       tournament_size=3,
                                       generations_number=10,
                                       n_jobs=ncpus,
                                       error_score=-1)
    cv.fit(*train)
    MODEL.set_params(**cv.best_params_)
    MODEL.fit(*train)
    test_score = MODEL.score(*test)
    outputs.append((test_score, tuple(cv.best_params_.items())))
outputs.sort()
pprint(outputs)
