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

class sea_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*81, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 64),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        probas = self.linear_relu_stack(x)
        return probas

class land_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*81, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 56),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        probas = self.linear_relu_stack(x)
        return probas

class builds_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*81, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        probas = self.linear_relu_stack(x)
        return probas

class RewardsCalculator:
    def __init__(self, territories, G, convoy_pairs, convoyable_countries):
        self.territories = territories
        self.G = G
        self.convoy_pairs = convoy_pairs
        self.convoyable_countries = convoyable_countries

    def calc_order_reward(self, units_df, territories_df, owner, target, old_territories_df, allies, resolved_orders):
        begin = time()
        target_control_reward = (territories_df[(territories_df['country']==target)&(territories_df['controlled_by']==owner)].shape[0] - old_territories_df[(old_territories_df['country']==target)&(old_territories_df['controlled_by']==owner)].shape[0])*2+1
        #2 and 1 are just arbitrary numbers here, can modify

        allied = units_df.loc[units_df['owner'].isin(allies[owner]), 'location'].values
        non_allied = units_df.loc[~units_df['owner'].isin(allies[owner]), 'location'].values
        allied_mean_dist = np.mean([nx.shortest_path_length(self.G, target, r) for r in allied])
        non_allied_mean_dist = np.mean([nx.shortest_path_length(self.G, target, r) for r in non_allied])
        mean_dist_ratio_reward = non_allied_mean_dist/allied_mean_dist

        lost_territories = units_df.loc[(units_df['location'].isin(old_territories_df.loc[old_territories_df['controlled_by']==owner, 'country'].values))&\
                 (units_df['location'].isin(territories_df.loc[territories_df['controlled_by']!=owner, 'country'].values))&\
                 (units_df['location']!=target), 'location'].values
        lost_territories_reward = -1*sum([10/nx.shortest_path_length(self.G, target, r) for r in lost_territories]) #10 is just an arbitrary number here, can modify

        controlled_scs = territories_df.loc[(territories_df['sc'])&(territories_df['sc_control']==owner), 'country'].values
        if len(controlled_scs) == 0:
            mean_sc_dist_ratio_reward = 0
        else:
            unallied_start, unallied_end = list(zip(*resolved_orders.loc[~resolved_orders['owner'].isin(allies[owner]), ['start', 'end']].values))
            unallied_mean_dist_start = np.mean([nx.shortest_path_length(self.G, l, r) for r in unallied_start for l in controlled_scs])
            unallied_mean_dist_end = np.mean([nx.shortest_path_length(self.G, l, r) for r in unallied_end for l in controlled_scs])
            mean_sc_dist_ratio_reward = unallied_mean_dist_end/unallied_mean_dist_start
        calc_time_diff(begin, 'calc_order_reward')
        return sum([target_control_reward, mean_dist_ratio_reward, lost_territories_reward, mean_sc_dist_ratio_reward])

    def calc_unit_performance_reward(self, units_df, c, unit_id, unit_rewards):
        begin = time()
        mean_unit_rewards = np.mean([d[unit_id]['reward'] for d in unit_rewards])
        owner_unit_ids = units_df.loc[units_df['owner']==c, 'unit_id'].tolist()
        mean_owner_rewards = np.mean([d[o_id]['reward'] for d in unit_rewards for o_id in owner_unit_ids])
        calc_time_diff(begin, 'calc_unit_performance_reward')
        return mean_owner_rewards - mean_unit_rewards

    def calc_owner_disband_unit_reward(self, total_build_rewards_grads):
        begin = time()
    #     print(total_build_rewards_grads)
        # subtract mean unit reward after and before disband for each country
        pre_dict = {c:[] for c in ['france','italy','england','russia','germany','austria','turkey']}
        post_dict = {c:[] for c in ['france','italy','england','russia','germany','austria','turkey']}
        for d in total_build_rewards_grads[:2]:
    #         print(d)
            for k, v in d.items():
                pre_dict[v['owner']].append(v['reward'])
        for d in total_build_rewards_grads[2:]:
            for k, v in d.items():
                post_dict[v['owner']].append(v['reward'])
    #     for m in [d for l in total_build_rewards_grads[:2] for d in l]:
    #         pre_dict[m['owner']].append(m['reward'])
    #     for m in [d for l in total_build_rewards_grads[2:] for d in l]:
    #         post_dict[m['owner']].append(m['reward'])
        calc_time_diff(begin, 'calc_owner_disband_unit_reward')
        return {c: np.mean(post_dict[c])-np.mean(pre_dict[c]) for c in ['france','italy','england','russia','germany','austria','turkey']}

    def discount_rewards(self, rewards, discount_factor):
        begin = time()
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * discount_factor
        calc_time_diff(begin, 'discount_rewards')
        return discounted

    def discount_and_normalize_rewards(self, all_rewards, discount_factor):
        begin = time()
        all_discounted_rewards = [self.discount_rewards(rewards, discount_factor)
                                  for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        calc_time_diff(begin, 'discount_and_normalize_rewards')
        return [(discounted_rewards - reward_mean) / reward_std
                for discounted_rewards in all_discounted_rewards]

    def discount_normalize_apply_build_rewards(self, all_build_rewards_grads, discount_factor):
        begin = time()
        #[{'france':{uid:{reward:value, grads:gradients}, uid:{reward:value, grads:gradients}}..., 'russia':{uid:{reward:value, grads:gradients}, uid:{reward:value, grads:gradients}}...}...{'france':{...}, 'russia':{}...}...]
        #take mean value for last set of rewards for country, apply discounted mean reward to all previous rewards for country, iterate
        flat_rewards = [v_2['reward'] for v_1 in all_build_rewards_grads[-1].values() for v_2 in v_1.values()]
        for step in range(len(all_build_rewards_grads)-2, -1, -1):
            mean_country_rewards = {k:np.mean([v_2['reward'] for v_2 in v.values()])*discount_factor for k, v in all_build_rewards_grads[step+1].items()}
            #{'france':last_step_average_discounted_reward, 'russia':last_step_average_discounted_reward}
            all_build_rewards_grads[step] = {k_1:
                                             {k_2:
                                              {'reward':v_2['reward']+mean_country_rewards[k_1],
                                               'grads':v_2['grads']}
                                              for k_2, v_2 in v_1.items()}
                                             for k_1, v_1 in all_build_rewards_grads[step].items()}
            flat_rewards.extend([v_2['reward'] for v_1 in all_build_rewards_grads[step].values() for v_2 in v_1.values()])
        reward_mean = np.mean(flat_rewards)
        reward_std = np.std(flat_rewards)
        reward_grads = []
        for i, d in enumerate(all_build_rewards_grads):
            for c, c_v in d.items():
                for u, u_v in c_v.items():
                    all_build_rewards_grads[i][c][u]['reward'] = (u_v['reward']-reward_mean)/reward_std
                    reward_grads.append([all_build_rewards_grads[i][c][u]['reward'] * g for g in all_build_rewards_grads[i][c][u]['grads']])
        calc_time_diff(begin, 'discount_normalize_apply_build_rewards')
        return reward_grads #all_build_rewards_grads


def make_model_input_arrays(owner, units_df, territories_df, move=False, unit=None, num_builds=0):
    begin = time()
    out_sub = territories_df['country'].to_frame().merge(units_df, how = 'left', left_on = 'country', right_on = 'location')[['country', 'owner', 'type', 'unit_id']].fillna('unoccupied')
    allies_array = ((out_sub['owner']==owner)&(out_sub['country']!=unit)).astype(int).tolist()
    enemies_array = (~(out_sub['owner'].isin([owner, 'unoccupied']))).astype(int).tolist()
    open_array = (out_sub['owner']=='unoccupied').astype(int).tolist()
    combined_array = [1 if a == 1 else 0 if o == 1 else -1 if e == 1 else 1 for a, e, o in zip(allies_array, enemies_array, open_array)]
#     unit_type_array = (out_sub['type']=='army').astype(int).tolist()
    unit_type_array = [1 if t == 'army' else -1 if t == 'fleet' else 0 for t in (out_sub['type']).tolist()]
    if move:
        active_unit_array = (out_sub['country']==unit).astype(int).tolist()
#         return active_unit_array, allies_array, enemies_array, open_array, unit_type_array
        calc_time_diff(begin, 'make_model_input_arrays')
        return active_unit_array, combined_array, unit_type_array
    else:
        if num_builds > 0:
            possible_choices = territories_df.loc[(territories_df['start_control']==owner)&\
                                                  (territories_df['controlled_by']==owner)&\
                                                  (territories_df['sc'])&\
                                                  (~territories_df['country'].str.split('_').str[0].isin(units_df['location'].str.split('_').str[0])),
                                                  'country'].tolist()
            build_disband_choices = out_sub['country'].isin(possible_choices).astype(int).tolist()
        elif num_builds < 0:
            build_disband_choices = allies_array
        else:
            build_disband_choices = [0]*out_sub.shape[0]
#         return build_disband_choices, allies_array, enemies_array, open_array, unit_type_array
        calc_time_diff(begin, 'make_model_input_arrays')
        return build_disband_choices, combined_array, unit_type_array


