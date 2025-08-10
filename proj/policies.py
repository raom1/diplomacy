import pandas as pd
from collections import Counter
from itertools import combinations
import numpy as np
import ast
import copy
import networkx as nx
import random
from uuid import uuid4
import math
from time import sleep, time

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn, optim

from utils.aux_funcs import calc_time_diff
from utils.rl_utils import make_model_input_arrays


def test_policy(target_territories_list, unit, owner, model, units_df, territories_df, loss_fn):
    begin = time()
    active_unit_array, combined_array, unit_type_array = make_model_input_arrays(owner, units_df, territories_df, move=True, unit=unit)
    if units_df[units_df['location']==unit]['metadata'].tolist()[0]["is_active"]:
        out_probs = model(torch.tensor(torch.tensor(np.array([active_unit_array, combined_array, unit_type_array])[np.newaxis]).float(), requires_grad=True))
    else:
        out_probs_shape = list(model.parameters())[-1].shape[0]
        out_probs = torch.tensor([[1/out_probs_shape]*out_probs_shape])
        
    rand_prob = torch.rand(1)
    cumsum = torch.cumsum(out_probs[0], dim=0)
    action = torch.where(torch.logical_and(cumsum[1:] > rand_prob, cumsum[:-1] < rand_prob) == True)
    if len(action[0]) == 0:
        if rand_prob < cumsum[0]:
            action = (torch.tensor([0], dtype=torch.int64),)
        elif rand_prob > cumsum[-1]:
            action = (torch.tensor([len(cumsum) - 1], dtype=torch.int64),)
    y_target = torch.zeros(1, len(out_probs[0]))
    try:
        y_target[0, action] = 1
    except TypeError:
        print(action)
        print(rand_prob)
        print(out_probs)
        print(cumsum)
        print(active_unit_array)
        print(combined_array)
        print(unit_type_array)
        raise TypeError
    if units_df[units_df['location']==unit]['metadata'].tolist()[0]["is_active"]:
        loss = loss_fn(out_probs, y_target)
        loss.backward()
        grads = [param.grad for param in model.parameters()]
    else:
        grads = [[]]
    target_territory = target_territories_list[action[0].numpy()[0]]
    calc_time_diff(begin, 'test_policy')
    return target_territory, out_probs, grads


def test_build_policy(num_builds, owner, model, units_df, build_loss_fn, territories_df):
    begin = time()
    build_disband_choices, combined_array, unit_type_array = make_model_input_arrays(owner, units_df, territories_df, move=False, num_builds=num_builds)
    # make the proportion of army units the action probability, rather than random uniform
    if num_builds < 0:
        try:
            unit_type_count = units_df[units_df['owner'] == owner]['type'].value_counts().sort_index()
            action_prob = unit_type_count['army']/sum(unit_type_count)
        except KeyError as e:
            action_prob = 0/sum(unit_type_count)
    elif num_builds > 0:
        available_build_types = territories_df.loc[(territories_df['start_control']==owner)&\
                                              (territories_df['controlled_by']==owner)&\
                                              (territories_df['sc'])&\
                                              (~territories_df['country'].str.split('_').str[0].isin(units_df['location'].str.split('_').str[0])),
                                                  'unit_type'].tolist()
        # commenting out for now, may need to add back later. Didn't work when calculating loss with pytorch
        # try:
        #     action_prob_offset = sum([t != 'both' for t in available_build_types])/len(available_build_types)
        # except ZeroDivisionError:
        #     action_prob_offset = 0
        # action_prob = torch.rand(1) + action_prob_offset
    # print(f"{units_df[units_df['owner']==owner]['metadata'].tolist()[0]['is_active']}")
    if units_df[units_df['owner']==owner]['metadata'].tolist()[0]["is_active"]:
        out_probs = model(torch.tensor(torch.tensor(np.array([build_disband_choices, combined_array, unit_type_array])[np.newaxis]).float(), requires_grad=True))
    else:
        out_probs = torch.tensor([[0.5, 0.5]])
    # print(out_probs)
    # out_probs += action_prob # also commented for switch to pytorch
    rand_prob = torch.rand(1)
    cumsum = torch.cumsum(out_probs[0], dim=0)
    try:
        action = torch.where(torch.logical_and(cumsum[1:] > rand_prob, cumsum[:-1] < rand_prob) == True)
    except IndexError:
        action = (torch.tensor([0], dtype=torch.int64),)
    if len(action[0]) == 0:
        if rand_prob < cumsum[0]:
            action = (torch.tensor([0], dtype=torch.int64),)
        elif rand_prob > cumsum[-1]:
            action = (torch.tensor([len(cumsum) - 1], dtype=torch.int64),)
    y_target = torch.zeros(1, len(out_probs[0]))
    y_target[0, action] = 1
    if units_df[units_df['owner']==owner]['metadata'].tolist()[0]["is_active"]:
        try:
            loss = build_loss_fn(out_probs, y_target)
        except RuntimeError:
            print(out_probs)
            print(y_target)
            raise RuntimeError
        loss.backward()
        grads = [param for name, param in model.named_parameters()]
    else:
        grads = [[]]
    build_disband_list = ['army', 'fleet']
    build_disband = build_disband_list[action[0].numpy()[0]]
    all_choices = territories_df[[bool(i) for i in build_disband_choices]]
    model_choices = all_choices.loc[territories_df['unit_type'].isin(['both', build_disband]), 'country'].tolist()
    if all_choices.shape[0] == 0:
        build_disband_choice = None
    elif len(model_choices) == 0: # and all_choices.shape[0] > 0
        assert build_disband not in units_df[units_df['owner']==owner]['type'].tolist(), '{} does have {}, check what went wrong'.format(owner, build_disband)
        #print('TO DO: Skip model prediction if only one unit type for owner')
        if build_disband == 'fleet':
            build_disband = 'army'
        else:
            build_disband = 'fleet'
        model_choices = all_choices.loc[territories_df['unit_type'].isin(['both', build_disband]), 'country'].tolist()
        build_disband_choice = random.choice(model_choices)
    else:
        build_disband_choice = random.choice(model_choices)
    calc_time_diff(begin, 'test_build_policy')
    return (build_disband_choice, build_disband), out_probs, grads #army_proba


def test_naive_policy(target_territories_list, output_dim):
    begin = time()
    out_probs = [1/output_dim]*output_dim
    grads = [[]]
    target_territory = random.choice(target_territories_list)
    calc_time_diff(begin, 'test_naive_policy')
    return target_territory, out_probs, grads


def test_naive_build_policy(num_builds, owner, model, units_df, build_loss_fn, territories_df):
    begin = time()
    if num_builds > 0:
        possible_builds = list(territories_df.loc[(territories_df['start_control']==owner)&\
                                                  (territories_df['controlled_by']==owner)&\
                                                  (territories_df['sc'])&\
                                                  (~territories_df['country'].str.split('_').str[0].isin(units_df['location'].str.split('_').str[0])),
                                                  ['country', 'unit_type']].itertuples(index=False, name=None))
        cleaned_possible_builds = []
        for b in possible_builds:
            if b[1] == 'both':
                cleaned_possible_builds.append((b[0], 'army'))
                cleaned_possible_builds.append((b[0], 'fleet'))
            elif b[0] == 'St Petersburg':
                cleaned_possible_builds.append(b)
                cleaned_possible_builds.append(('St Petersburg_nc', 'fleet'))
                cleaned_possible_builds.append(('St Petersburg_sc', 'fleet'))
            else:
                cleaned_possible_builds.append(b)
        try:
            to_build = random.choice(cleaned_possible_builds)
        except IndexError:
            to_build = (None, None)
        calc_time_diff(begin, 'test_naive_build_policy')
        return to_build, [1/territories_df.shape[0]]*territories_df.shape[0], [[]]
    elif num_builds < 0:
        units_sub = units_df.loc[units_df['owner']==owner, 'location'].tolist()
        calc_time_diff(begin, 'test_naive_build_policy')
        return (random.choice(units_sub), 'army'), [1/territories_df.shape[0]]*territories_df.shape[0], [[]] #disbands


