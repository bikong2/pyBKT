# -*-coding: utf-8 -*-
# @author: xihuali
# @date:   2020.02.24
# @brief:  bkt

import numpy as np
import sys
sys.path.append('../')
from pyBKT.generate import synthetic_data
from pyBKT.generate import random_model, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from pyBKT.util import print_dot
from copy import deepcopy

#parameters
# num_subparts: The number of unique questions used to check understanding
# num_resources: The number of unique learning resources available to students
num_subparts = 20
num_resources = 1 # (r1,...)
num_fit_initializations = 5

# num_students * num_observations_each
observation_sequence_lengths = np.full(3, 3, dtype=np.int)
p_T = 0.30 # transit
p_F = 0.00 # forget
p_G = 0.10 # guess
p_S = 0.03 # slip
p_L0 = 0.10

#generate synthetic model and data.
truemodel = {}

truemodel["As"] = np.zeros((num_resources, 2, 2), dtype=np.float_)
for i in range(num_resources):
    truemodel["As"][i, :, :] = np.transpose([[1-p_T, p_T], [p_F, 1-p_F]])
truemodel["learns"] = truemodel["As"][:, 1, 0]
truemodel["forgets"] = truemodel["As"][:, 0, 1]

truemodel["pi_0"] = np.array([[1-p_L0], [p_L0]])
truemodel["prior"] = truemodel["pi_0"][1][0]

truemodel["guesses"] = np.full(num_subparts, p_G, dtype=np.float_)
truemodel["slips"] = np.full(num_subparts, p_S, dtype=np.float_)

# random the sequential id of the resources at each checkpoint
truemodel["resources"] = np.random.randint(1, high = num_resources+1, size = sum(observation_sequence_lengths))

#data
print("generating data...")
#data = synthetic_data.synthetic_data(truemodel, observation_sequence_lengths, truemodel["resources"])
data = synthetic_data.synthetic_data(truemodel, observation_sequence_lengths)
print(data['data'])
print(data['data'].shape)

#fit models, starting with random initializations
print('fitting! each dot is a new EM initialization')

best_likelihood = float("-inf")

for i in range(num_fit_initializations):
    print_dot.print_dot(i, num_fit_initializations)
    fitmodel = random_model.random_model(num_resources, num_subparts)
    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
    print(log_likelihoods[-1][-1])
    if (log_likelihoods[-1] > best_likelihood):
        best_likelihood = log_likelihoods[-1]
        best_model = fitmodel

# compare the fit model to the true model
print('')
print('these two should look similar: As...')
print(truemodel['As'])
print('')
print(best_model['As'])

print('')
print('these should look similar too: guesses...')
print(1-truemodel['guesses'])
print('')
print(1-best_model['guesses'])

print('')
print('these should look similar too: slips..')
print(1-truemodel['slips'])
print('')
print(1-best_model['slips'])

print('')
print('................prediction')
(correct_predictions, state_predictions) = predict_onestep.run(best_model, data)
for i in range(len(correct_predictions)):
    print("time i=%d" % i, correct_predictions)
print(state_predictions)

