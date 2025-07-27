import os
import pickle
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
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
from utils.rl_utils import * #sea_nn, land_nn, builds_nn
from move_resolver import MoveResolver
from builds_resolver import BuildsMaker
from policies import test_policy, test_build_policy, test_naive_policy, test_naive_build_policy

step_timing_dict = {}
make_orders_timing_dict_list = []

all_num_rounds = []
active_countries_list = []
winners = []
end_sc_state = []

# all_save_grads_list = []
# all_save_build_grads_list = []

torch.manual_seed(42)

device = 'cpu'

apply_rl = True

sea_model = sea_nn().to(device)

land_model = land_nn().to(device)

builds_model = builds_nn().to(device)

sea_optimizer = optim.NAdam(sea_model.parameters())
land_optimizer = optim.NAdam(land_model.parameters())
loss_fn = nn.BCELoss()

build_optimizer = optim.NAdam(builds_model.parameters())
build_loss_fn = nn.BCELoss()

active_country = "italy"

def initialize_game_assets():
	if not os.path.isdir("assets/game_assets"):
		from utils.game_setup import create_game_assets
		create_game_assets()

	pkl_assets = []
	for a in ["diplomacy_graph", "territories", "convoy_pairs", "convoyable_countries"]:
		# print(a)
		with open(f"assets/game_assets/{a}.pkl", "rb") as file:
			pkl_assets.append(pickle.load(file))

	G, territories, convoy_pairs, convoyable_countries = pkl_assets

	territories_df = pd.read_csv("assets/game_assets/territories_df.csv")
	territories_df["connected_to"] = territories_df["connected_to"].apply(ast.literal_eval)
	# territories_df["coords"] = territories_df["coords"].apply(ast.literal_eval)
	units_df = pd.read_csv("assets/game_assets/units_df.csv")

	return territories, territories_df, units_df, G, convoy_pairs, convoyable_countries

def play_one_round(policy, units_df, territories_df, o = None, retreat = False, season = None, human = False, model = None, targets = None, move_resolver = None, allies = None):
    begin = time()
    if not retreat:
        assert season in [None, 'spring', 'fall'], '{} not a valid value for season'.format(season)
        if season is None:
            season = 'spring'
        elif season == 'spring':
            season = 'fall'
        else:
            season = 'spring'        
    if human:
        o = input('input all orders: ')
        o = ast.literal_eval(o)
    resolved_orders, original_orders = move_resolver.resolve_submitted_moves(units_df, territories_df, orders_list=o)
    resolved_orders['target'] = targets
    original_orders['target'] = targets
    if not retreat:
        need_retreat = resolved_orders[resolved_orders['dislodged']]
        for ind, s in need_retreat['start'].items():
            try:
                outcome, (start, order, end, support, convoy) = move_resolver.check_order(units_df, territories_df, s, 'retreat')
                if order == 'disband':
                    resolved_orders.loc[resolved_orders['start']==s, 'order'] = 'disband'
                    need_retreat = need_retreat.drop(ind)
            except Exception as e:
                continue
        if len(need_retreat) > 0:
            if human:
                print('Units that need to retreat:')
                for c in need_retreat['start'].tolist():
                    print(c)
                # TO DO: Show updated map to help with retreats
                r = input('Enter orders for retreats: ')
                r = ast.literal_eval(r)
            else:
                r = []
                for c in ['france','italy','england','russia','germany','austria','turkey']:
                    orders_list, _, _, _, _, _ = move_resolver.make_orders(units_df, territories_df, test_policy, loss_fn, sea_model=sea_model, land_model=land_model, allies=allies, naive_policy=test_naive_policy, active_country=active_country)
                    r.extend(orders_list)
                drop_inds = []
                for ind, (s, o, e) in enumerate(r):
                    if o == 'disband':
                        resolved_orders.loc[resolved_orders['start']==s, 'order'] = 'disband'
                        drop_inds.append(ind)
                r = [m for i, m in enumerate(r) if i not in drop_inds]
                assert not any([o == 'disband' for s, o, e in r]), 'still have some disband orders: {}'.format(r)
            resolved_retreats = play_one_round(policy, units_df, territories_df, r, retreat=True, season=season, move_resolver=move_resolver, allies = allies)
            resolved_retreats.loc[resolved_retreats['success']==False, 'order'] = 'disband'
            resolved_retreats.loc[resolved_retreats['order']=='move', 'order'] = 'retreat'
            for c in resolved_retreats['start']:
                resolved_orders.loc[resolved_orders['start']==c, ['start', 'order', 'end', 'support', 'convoy', 'count']] = resolved_retreats.loc[resolved_retreats['start']==c, ['start', 'order', 'end', 'support', 'convoy', 'count']].values[0].tolist()
        units_df = resolved_orders[resolved_orders['order'] != 'disband'][['end', 'type', 'owner', 'unit_id']]
        units_df.columns = ['location', 'type', 'owner', 'unit_id']
        calc_time_diff(begin, 'play_one_round')
        return units_df, resolved_orders, original_orders, season, *move_resolver.update_territories_df(units_df, territories_df, season)
    else:
        calc_time_diff(begin, 'play_one_round')
        return resolved_orders

def check_win(territories_df, rounds):
    begin = time()
    sc_count = territories_df[(territories_df['sc'])&(territories_df['coast']==False)]['sc_control'].value_counts()
    for c, scs in sc_count.items():
        if (scs >= 18) or (rounds >= 350 and scs == sc_count.max()):
            return c
    calc_time_diff(begin, 'check_win')
    return None


def main():
	for _ in range(10): #100
		print('starting batch {}'.format(_))
		num_rounds = []

		discount_factor = 0.5
		build_discount_factor = 0.5
		num_trials = 1

		if active_country is None:
			country_iter_list = ['france','italy','england','russia','germany','austria','turkey']
		else:
			country_iter_list = [active_country]

		for _ in range(num_trials):
			territories, territories_df, units_df, G, convoy_pairs, convoyable_countries = initialize_game_assets()
			move_resolver = MoveResolver(territories, G, convoy_pairs, convoyable_countries)
			rewards_calculator = RewardsCalculator(territories, G, convoy_pairs, convoyable_countries)
			builds_maker = BuildsMaker(territories, G, convoy_pairs, convoyable_countries)
			total_rewards_grads = []
			total_build_rewards_grads = []
			all_build_rewards_grads = []
			calc_build_rewards = False
			season = None
			winner = None
			rounds = 0
			save_grads_list = []
			save_build_grads_list = []
			while winner is None and rounds < 350:
				rounds += 1
				print(f"starting round {rounds}")
				allies = {country:[country] for country in ['france','italy','england','russia','germany','austria','turkey']}
				sea_optimizer.zero_grad()
				land_optimizer.zero_grad()
				build_optimizer.zero_grad()
				all_orders, all_out_probs, all_grads, targets_list, all_model_vers, active_country_list = move_resolver.make_orders(units_df, territories_df, test_policy,
																									loss_fn,
																									sea_model=sea_model,
																									land_model=land_model,
																									allies=allies,
																									naive_policy=test_naive_policy,
																									active_country=active_country) #test_policy
				units_df["metadata"] = [{"is_active": a, "out_probs_ind": p, "grads_ind": g, "model_ver": v}for a, p, g, v in zip(active_country_list, range(len(all_out_probs)), range(len(all_grads)), all_model_vers)]
				units_df, resolved_orders, original_orders, season, old_territories_df, territories_df = play_one_round(test_policy,
																														units_df,
																														territories_df,
																														o=all_orders,
																														season=season,
																														targets=targets_list,
																														move_resolver=move_resolver,
																														allies=allies
																														)
				assert not any([v > 1 for v in Counter([c.split('_')[0] for c in units_df['location']]).values()]), 'too many units in a country, check units_df'
				
				## CALCULATE MOVE REWARDS ##
				if apply_rl:
					all_rewards_grads = {uid:{'reward':0, 'grads':np.nan, 'owner':None, 'model_ver':None} for owner, uid in resolved_orders[["owner", "unit_id"]].itertuples(index=None, name=None) if owner == active_country} #'reward':np.nan
					# for grads, ver, (ind, row) in zip(all_grads, all_model_vers, resolved_orders.iterrows()):
					for ind, row in resolved_orders.iterrows():
						if row["metadata"]['is_active']:
							grads = all_grads[row["metadata"]['grads_ind']]
							ver = row["metadata"]['model_ver']
							if grads == [[]]:
								print(row)
								raise
							if row['owner'] is None:
								print(row)
							if row['owner'] == active_country:
								# print(row['owner'], active_country)
								if math.isnan(rewards_calculator.calc_order_reward(units_df, territories_df, row['owner'], row['target'], old_territories_df, allies, resolved_orders)):
									print('got nan reward:')
									print(row)
									raise Exception('nan reward')
								all_rewards_grads[row['unit_id']]['reward'] = rewards_calculator.calc_order_reward(units_df, territories_df, row['owner'], row['target'], old_territories_df, allies, resolved_orders)
								all_rewards_grads[row['unit_id']]['grads'] = grads
								all_rewards_grads[row['unit_id']]['owner'] = row['owner']
								all_rewards_grads[row['unit_id']]['model_ver'] = ver
					# share reward across allied units, adjusted by distance. Only do if active_country is None
					if active_country is None:
						for owner, df in list(resolved_orders.groupby('owner')):
							combos = combinations(df['end'].tolist(), 2)
							for l, r in combos:
								dist = nx.shortest_path_length(G, l, r) #+ 0.0001 #ensure no div by zero
								l_reward = all_rewards_grads[df.loc[df['end']==l, 'unit_id'].values[0]]['reward']
								r_reward = all_rewards_grads[df.loc[df['end']==r, 'unit_id'].values[0]]['reward']
								all_rewards_grads[df.loc[df['end']==l, 'unit_id'].values[0]]['reward'] += r_reward*(0.5/dist)
								all_rewards_grads[df.loc[df['end']==r, 'unit_id'].values[0]]['reward'] += l_reward*(0.5/dist)
					total_rewards_grads.append(all_rewards_grads)
					total_build_rewards_grads.append(all_rewards_grads)
				if season == 'fall':
					if not calc_build_rewards:
						if apply_rl:
							calc_build_rewards = True
					else:
						## CALCULATE BUILD REWARDS ##
						assert len(total_build_rewards_grads) == 4, 'Incorrect number of observations to calculate build rewards. Assumed 4, found {}'.format(len(total_build_rewards_grads))
						if active_country is None:
							pre_build = {k:0 for k in set([v['owner'] for v in total_build_rewards_grads[0].values()])}
						else:
							pre_build = {active_country: 0}
						for v in total_build_rewards_grads[1].values():
							pre_build[v['owner']] += 1
						post_build = Counter([v['owner'] for v in total_build_rewards_grads[2].values()])
						for k in pre_build.keys():
							if k not in post_build.keys():
								post_build[k] = 0
						pre_post_diff = {k: post_build[k] - pre_build[k] for k in pre_build.keys()}
						num_builds_owner = {}
						for c in country_iter_list:
							num_builds_owner[c] = sum((territories_df['sc_control']==c)&(territories_df['sc']==True)&(territories_df['coast']==False)) - sum(units_df['owner']==c)
						owner_build_disband_mean_reward = rewards_calculator.calc_owner_disband_unit_reward(total_build_rewards_grads)
						for k, v in builds_output_dict.items():
							if pre_post_diff[v['owner']] > 0:
								# why doing this? it seems like a very high reward, changed from 5 to 2 on 4/20/24
								builds_output_dict[k]['rewards'].append(num_builds_owner[total_build_rewards_grads[2][k]['owner']]*2)
								builds_output_dict[k]['rewards'].append(sum(d[k]['reward'] for d in total_build_rewards_grads[2:] if k in d.keys()))
							elif pre_post_diff[v['owner']] < 0:
								try:
									builds_output_dict[k]['rewards'].append((num_builds_owner[total_build_rewards_grads[1][k]['owner']]+1)*2)
									builds_output_dict[k]['rewards'].append(owner_build_disband_mean_reward[total_build_rewards_grads[1][k]['owner']])
								except KeyError:
									if k in total_build_rewards_grads[2].keys() and k in total_build_rewards_grads[3].keys():
										builds_output_dict[k]['rewards'].append(num_builds_owner[total_build_rewards_grads[2][k]['owner']]*2)
										builds_output_dict[k]['rewards'].append(sum(d[k]['reward'] for d in total_build_rewards_grads[2:]))
						builds_output_dict = {k:{'reward': sum(v['rewards']), 'grads': v['grads'], 'owner': v['owner']} for k, v in builds_output_dict.items()}
						grouped_build_output_dict = {c:{} for c in country_iter_list}
						for k, v in builds_output_dict.items():
							grouped_build_output_dict[v['owner']][k] = {'reward': v['reward'], 'grads': v['grads']}
						all_build_rewards_grads.append(grouped_build_output_dict)
						total_build_rewards_grads = total_build_rewards_grads[2:]
						# first make model for builds
					units_df, builds_output_dict = builds_maker.make_builds(units_df,
																			territories_df,
																			build_loss_fn,
																			policy = test_build_policy,
																			model = builds_model,
																			unit_rewards = total_build_rewards_grads,
																			active_country = active_country,
																			naive_builds_policy = test_naive_build_policy,
																			rewards_calculator = rewards_calculator,
																			country_iter_list = country_iter_list) #test_build_policy
					save_build_grads_list.append(builds_output_dict)

				winner = check_win(territories_df, rounds)
				if winner is not None:
					print('{} is the winner!'.format(winner))
					for uid, v in total_rewards_grads[-1].items():
						if v['owner'] == winner and territories_df[territories_df['country'] == units_df[units_df['unit_id'] == uid]['location'].values[0]]['sc'].values[0]:
							total_rewards_grads[-1][uid]['reward'] += 100
					winners.append(winner)
					end_sc_state.append(territories_df[(territories_df['sc'])&(~territories_df['coast'])]['sc_control'].value_counts())
			num_rounds.append(rounds)
			## UPDATE MOVE GRADIENTS ##
			# print(total_rewards_grads)
			unit_rewards_dict = {k:{'rewards':[], 'grads':[], 'model_ver':[]} for k in list(set([k for l in [list(d.keys()) for d in total_rewards_grads] for k in l]))}
			for d in total_rewards_grads:
				for k, v in d.items():
					unit_rewards_dict[k]['rewards'].append(v['reward'])
					unit_rewards_dict[k]['grads'].append(v['grads'])
					unit_rewards_dict[k]['model_ver'].append(v['model_ver'])
			# orders model update
			all_rewards = []
			all_grads = []
			all_model_ver = []
			# print(all_grads)
			# print(unit_rewards_dict)
			for _, v in unit_rewards_dict.items():
				all_rewards.append(v['rewards'])
				all_grads.extend(v['grads'])
				all_model_ver.extend(v['model_ver'])
			all_rewards = [r for a in rewards_calculator.discount_and_normalize_rewards(all_rewards, discount_factor) for r in a]
			sea_all_rewards_flat = []
			land_all_rewards_flat = []
			sea_all_grads = []
			land_all_grads = []
			# print(all_rewards)
			# print(all_grads)
			for ind, v in enumerate(all_model_ver):
				# print(ind, v)
				if v == 'sea':
					sea_all_rewards_flat.append(all_rewards[ind])
					sea_all_grads.append(all_grads[ind])
				elif v == 'land':
					land_all_rewards_flat.append(all_rewards[ind])
					land_all_grads.append(all_grads[ind])
			###testing
			# for r, grads in zip(sea_all_rewards_flat, sea_all_grads):
			# 	print(r)
			# 	for g in grads:
			# 		print(len(g))
			# 		print([g_2 for g_2 in g])
			# print('++++++++')
			# for r, grads in zip(land_all_rewards_flat, land_all_grads):
			# 	print(r)
			# 	for g in grads:
			# 		print(len(g))
			# 		print([g_2.shape for g_2 in g])
			###end testing
			sea_all_grad_rewards = list(zip(*[[r * g for g in grads] for r, grads in zip(sea_all_rewards_flat, sea_all_grads)]))
			land_all_grad_rewards = list(zip(*[[r * g for g in grads] for r, grads in zip(land_all_rewards_flat, land_all_grads)]))

			####land model is being applied to some sea units####

			sea_all_grad_rewards = [torch.mean(torch.stack(grads_list), 0) for grads_list in sea_all_grad_rewards]

			land_all_grad_rewards = [torch.mean(torch.stack(grads_list), 0) for grads_list in land_all_grad_rewards]

			for ind, param in enumerate(sea_model.parameters()):
				param.grad = sea_all_grad_rewards[ind]
			for ind, param in enumerate(land_model.parameters()):
				param.grad = land_all_grad_rewards[ind]
			try:
				sea_optimizer.step()
			except IndexError as e:
				assert(len(list(zip(sea_all_grad_rewards, sea_model.parameters())))) == 0, "something happened: sea_all_grad_rewards: {}, zip: {}, len trainable_variables: {}".format(sea_all_grad_rewards, len(list(zip(sea_all_grad_rewards, sea_model.parameters()))), len(sea_model.parameters()))
				print('no sea moves to update, this is an error, check what is happening')
			try:
				land_optimizer.step()
			except IndexError as e:
				assert(len(list(zip(land_all_grad_rewards, land_model.parameters())))) == 0, "something happened: land_all_grad_rewards: {}, zip: {}, len trainable_variables: {}".format(land_all_grad_rewards, len(list(zip(land_all_grad_rewards, land_model.parameters()))), len(land_model.parameters()))
				print('no land moves to update, this is an error, check what is happening')
			## UPDATE BUILD GRADIENTS ##
			build_reward_grads = rewards_calculator.discount_normalize_apply_build_rewards(all_build_rewards_grads, build_discount_factor)
			build_reward_grads = list(zip(*build_reward_grads))
			build_reward_grads = [torch.mean(torch.stack(grads_list), 0) for grads_list in build_reward_grads]
			try:
				build_optimizer.step()
			except IndexError as e:
				assert len(list(zip(build_reward_grads, builds_model.parameters()))) == 0, "something happened: build_reward_grads: {}, zip: {}, len trainable_variables: {}".format(build_reward_grads, len(list(zip(build_reward_grads, builds_model.parameters()))), len(builds_model.parameters()))
				print('no builds to update, this is an error, check what is happening')
		all_num_rounds.append(num_rounds)
		active_countries_list.append(active_country)
	print('done')

if __name__ == "__main__":

	main()


