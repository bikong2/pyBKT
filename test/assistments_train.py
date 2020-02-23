# -*- coding: utf-8 -*-
# @author: xihuali

import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni, random_model
from pyBKT.fit import EM_fit
from pyBKT.util.data_helper import assistments_data
from copy import deepcopy

skill = 'Pythagorean Theorem'

#parameters classes
num_resources = 1
num_fit_initializations = 20

#alternatively, you can load REAL data - might take a minute (involves dataset download each time the function is called)
print('loading assistments %s data...' % skill)
data = assistments_data(skill)
print(data['data'])
print(data['data'].shape)

#fit models, starting with random initializations
print('fitting! each dot is a new EM initialization')

best_likelihood = float("-inf")

for i in range(num_fit_initializations):
    fitmodel = random_model.random_model(num_resources, data['data'].shape[0]) # include this line to randomly set initial param values
    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
    print("%d steps: %f" % (i, log_likelihoods[-1]))
    if(log_likelihoods[-1] > best_likelihood):
        best_likelihood = log_likelihoods[-1]
        best_model = fitmodel

# compare the fit model to the true model

print('')
print('\tlearned')
print('prior\t%.4f' % (best_model["pi_0"][1][0]))
for r in range(num_resources):
    print('learn%d\t%.4f' % (r+1, best_model['As'][r, 1, 0].squeeze()))
for r in range(num_resources):
    print('forget%d\t%.4f' % (r+1, best_model['As'][r, 0, 1].squeeze()))

for s in range(num_resources):
    print('guess%d\t%.4f' % (s+1, best_model['guesses'][s]))
for s in range(num_resources):
    print('slip%d\t%.4f' % (s+1, best_model['slips'][s]))
