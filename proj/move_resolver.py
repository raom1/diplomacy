import pandas as pd
from collections import Counter
from itertools import combinations
import numpy as np
import ast
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import copy
import networkx as nx
import random
from uuid import uuid4
import math
from time import sleep, time
import pickle
import asyncio

from utils.aux_funcs import calc_time_diff

class MoveResolver:
	def __init__(self, territories, G, convoy_pairs, convoyable_countries):
		self.territories = territories
		self.G = G
		self.convoy_pairs = convoy_pairs
		self.convoyable_countries = convoyable_countries

	def check_destination_valid(self, start, end):
		begin = time()
		out = end in self.territories[start]['connected_to'] or (start, end) in self.convoy_pairs
		calc_time_diff(begin, 'check_destination_valid')
		return out

	def check_unit_type_valid(self, territories_df, unit_type, end):
		begin = time()
		out = unit_type == territories_df.loc[territories_df['country']==end, 'unit_type'].values[0] or territories_df.loc[territories_df['country']==end, 'unit_type'].values[0] == 'both'
		calc_time_diff(begin, 'check_unit_type_valid')
		return out

	def check_order(self, units_df, territories_df, start, order = None, end = None, support = None, convoy = None):

		begin = time()
		if order is None:
			order = 'hold'
		if end is None:
			end = start
		
		assert start in self.territories.keys(), '"{}" is not a recognized country'.format(start)
		assert end in self.territories.keys(), '"{}" is not a recognized country'.format(end)
		assert order in ['move', 'hold', 'support', 'convoy', 'retreat', 'disband'], '"{}" is not a valid order'.format(order)
		if support is not None:
			for c in support:
				assert c in self.territories.keys(), '"{}" is not a recognized country, cannot support'.format(c)
		if convoy is not None:
			for c in convoy:
				assert c in self.territories.keys(), '"{}" is not a recognized country, cannot convoy'.format(c)
		
		try:
			unit_type = units_df.loc[units_df['location']==start, 'type'].values[0]
		except IndexError:
			raise AssertionError('no unit in {}'.format(start))
		
		if order == 'move':
			assert self.check_unit_type_valid(territories_df, unit_type, end), 'unit in {} cannot move to {}'.format(start, end)
			assert self.check_destination_valid(start, end) or (self.territories[start]['unit_type'] == 'both' and self.territories[end]['unit_type'] == 'both'), '{} not connected to {}'.format(start, end)
			if unit_type == 'fleet':
				# ensure don't move from fleet/both to fleet/both if not connected by fleet/both territory
				assert len(set([t for t in self.territories[start]['connected_to'] if self.territories[t]['unit_type'] in ['fleet', 'both']])&set([t for t in self.territories[end]['connected_to'] if self.territories[t]['unit_type'] in ['fleet', 'both']])) > 0, 'cannot move fleet from {} to {}'.format(start, end)
		
		if order == 'support':
			if len(support) == 1:
				support = support * 2
			if support[0] == support[1]:
				support_move = 'hold'
			else:
				support_move = 'move'
			assert self.check_order(units_df, territories_df, start, 'move', support[1]) and self.check_order(units_df, territories_df, support[0], support_move, support[1])[0], 'cannot support {} or supported move is not valid'.format(support[1])
		
		if order == 'convoy':
			assert unit_type == 'fleet' and self.territories[start]['unit_type'] == 'fleet' and not self.territories[start]['coast']
			assert self.check_order(units_df, territories_df, convoy[0], 'move', convoy[1])[0]
			
		if order == 'retreat':
	#         print('got to retreat')
			if all([c in units_df['location'].tolist() for c in self.territories[start]['connected_to']]):
	#             print('caught')
				order = 'disband'
				end = np.nan
			else:
	#             print('missed')
				assert self.check_unit_type_valid(territories_df, unit_type, end), 'unit in {} cannot move to {}'.format(start, end)
				assert self.check_destination_valid(start, end) or (self.territories[start]['unit_type'] == 'both' and self.territories[end]['unit_type'] == 'both'), 'unit cannot move from {} to {}'.format(start, end)
		calc_time_diff(begin, 'check_order')
		return True, (start, order, end, support, convoy)

	def submit_move(self, units_df, territories_df, start, order = None, end = None, support = None, convoy = None):
		begin = time()
		valid, (start, order, end, support, convoy) = self.check_order(units_df, territories_df, start, order, end, support, convoy)
		assert valid
		calc_time_diff(begin, 'submit_move')
		return start, order, end, support, convoy

		# DEPRICATED?
	def cut_support(self, df):
		begin = time()
		df_sub = df.merge(df[df['order'] == 'move'], left_on = 'start', right_on = 'end', how = 'left')
		df.loc[(~df_sub['start_y'].isna())&(df['order']=='support'), 'order'] = 'hold'
		calc_time_diff(begin, 'cut_support')
		return df

	def count_support(self, df):
	#     print('counting support')
		begin = time()
		endpoints = df['end'].value_counts()
		resolve_moves = df[df['end'].isin(endpoints[endpoints > 1].index.tolist())]
		resolve_moves['count'] = 0
	#     print(resolve_moves)
		for ind, sup in df['support'].items():
			if sup is not None:
				s, e = sup
				resolve_moves.loc[(resolve_moves['start']==s)&(resolve_moves['end']==e), 'count'] += 1
		calc_time_diff(begin, 'count_support')
		return resolve_moves

	def detect_self_loops(self, submitted_moves_df):
		begin = time()
		move_sub = submitted_moves_df[submitted_moves_df['order']=='move']
		if move_sub.shape[0] > 0:
			move_sub['start'] = move_sub['start'].str.split('_').str[0]
			move_sub['end'] = move_sub['end'].str.split('_').str[0]
			DG = nx.DiGraph()
			DG.add_edges_from(list(zip(move_sub['start'].tolist(), move_sub['end'].tolist())))
			calc_time_diff(begin, 'detect_self_loops')
			return list(nx.simple_cycles(DG))
		else:
			calc_time_diff(begin, 'detect_self_loops')
			return []

	def check_successful_order(self, units_df, territories_df, submitted_moves_df, start, end, success = None, self_loops = []):
		begin = time()
		start_coast = start.split('_')[0]
		end_coast = end.split('_')[0]
		self_loop = ([l for l in self_loops if start in l]+[[]])[0]
		if success is None:
			## don't resolve previous moves of self loop to prevent infinite recursion ##
			if not (start_coast in self_loop and end_coast in self_loop):#any([start in l and end in l for l in self_loops]):
				## resolve dependent moves
				if end_coast in submitted_moves_df['start'].str.split('_').str[0].tolist() and submitted_moves_df.loc[submitted_moves_df['start'].str.split('_').str[0] == end_coast, 'order'].values[0] == 'move':
					rerun_start, rerun_end, rerun_success = submitted_moves_df[submitted_moves_df['start'].str.split('_').str[0] == end_coast][['start', 'end', 'success']].iloc[0].tolist()
					submitted_moves_df = self.check_successful_order(units_df, territories_df, submitted_moves_df, rerun_start, rerun_end, rerun_success, self_loops)
			## check convoy moves ##
			if end not in territories_df.loc[territories_df['country'] == start, 'connected_to'].values[0]+[start]:
				convoying_fleets = submitted_moves_df.loc[(submitted_moves_df['order'] == 'convoy')&(submitted_moves_df['convoy'].str[0]==start)&(submitted_moves_df['convoy'].str[1]==end), 'start'].tolist()
				for f in convoying_fleets:
					rerun_start, rerun_end, rerun_success = submitted_moves_df[submitted_moves_df['start'] == f][['start', 'end', 'success']].iloc[0].tolist()
					submitted_moves_df = self.check_successful_order(units_df, territories_df, submitted_moves_df, rerun_start, rerun_end, rerun_success, self_loops)
				convoying_fleets = submitted_moves_df[(submitted_moves_df['start'].isin(convoying_fleets))&(submitted_moves_df['success'])]['start'].tolist()
				G_sub = nx.Graph(self.G.subgraph(convoying_fleets+[start, end]))
				if G_sub.has_edge(start, end):
					G_sub.remove_edge(c_1, c_2)
				if not nx.has_path(G_sub, start, end):
					submitted_moves_df.loc[submitted_moves_df['start'] == start, ['success', 'order', 'end', 'count']] = [False, 'hold', start, 0]
					submitted_moves_df = self.check_successful_order(units_df, territories_df, submitted_moves_df, start, start, False, self_loops)
			## check support moves valid ##
			if submitted_moves_df.loc[submitted_moves_df['start'] == start, 'order'].values[0] == 'support':
				support_start, support_end = submitted_moves_df.loc[submitted_moves_df['start'] == start, 'support'].values[0]
				support_df = submitted_moves_df[(submitted_moves_df['start']==support_start)&(submitted_moves_df['end']==support_end)]
				if support_df.shape[0] == 0:
					submitted_moves_df.loc[submitted_moves_df['start'] == start, ['success', 'order', 'count']] = [False, 'hold', 0]
					submitted_moves_df = self.check_successful_order(units_df, territories_df, submitted_moves_df, start, start, False, self_loops)
				elif support_df.shape[0] > 1:
					raise IndexError('Same move multiple times, {} supporting both moves'.format(start))
			## Can't dislodge self ##
			if end_coast in units_df['location'].str.split('_').str[0].tolist() and \
				units_df.loc[units_df['location'].str.split('_').str[0] == end_coast, 'owner'].values[0] == units_df.loc[units_df['location'].str.split('_').str[0] == start_coast, 'owner'].values[0] and \
				end != start \
				and not submitted_moves_df.loc[submitted_moves_df['start'].str.split('_').str[0] == end_coast, 'success'].values[0]:
				try:
					assert submitted_moves_df.loc[submitted_moves_df['start'].str.split('_').str[0] == end_coast, 'order'].values[0] != 'move' or \
							(start_coast in self_loop and end_coast in self_loop and submitted_moves_df.loc[submitted_moves_df['start'].str.split('_').str[0]==end_coast, 'end'].values[0].split('_')[0] in self_loop)
				except AssertionError as e:
					print(start_coast)
					print(end_coast)
					print(submitted_moves_df.loc[submitted_moves_df['start'].str.split('_').str[0] == end_coast])
					print(self_loop)
					raise AssertionError(e)
				submitted_moves_df.loc[submitted_moves_df['start'] == start, ['success', 'order', 'end', 'count']] = [False, 'hold', start, 0]
				submitted_moves_df = self.check_successful_order(units_df, territories_df, submitted_moves_df, start, start, False, self_loops)
			else:
				resolve_moves = submitted_moves_df[(submitted_moves_df['end'].isin([end, end_coast, end_coast+'_sc', end_coast+'_nc', end_coast+'_ec']))|\
													(submitted_moves_df['start'].str.split('_').str[0].isin(self_loop))]
				rest = submitted_moves_df.drop(resolve_moves.index)
				not_moving = resolve_moves[resolve_moves['order']!='move']
				if not_moving.shape[0]>0:
					hold_owner = units_df[units_df['location']==not_moving['start'].values[0]]['owner'].values[0]
					m=(resolve_moves['order']=='move')&\
						(resolve_moves['start'].isin(units_df[units_df['owner']==hold_owner]['location'].tolist()))
					resolve_moves.loc[m,'success']=False
				if resolve_moves.shape[0] > 0 and resolve_moves['count'].value_counts().sort_index(ascending=False).tolist()[0]==1:
					maximum = resolve_moves['count'].max()
					top_count = maximum
				else:
					maximum = -1
					top_count = resolve_moves['count'].max()
				for ind, row in resolve_moves.iterrows():
					if row['success'] is None:
						if row['count'] == maximum and resolve_moves[(resolve_moves['start']==row['end'])&(resolve_moves['owner']==row['owner'])].shape[0] == 0:
							# catch edge case with self-loop and supported move into country with same owner ^^
							success = True
						elif row['order'] == 'convoy' and row['count'] == top_count:
							success = True
						else:
							success = False
						resolve_moves.loc[resolve_moves['start'] == row['start'], 'success'] = success
					else:
						success = row['success']
					if not success:
						resolve_moves.loc[resolve_moves['start'] == row['start'], 'order'] = 'hold'
						resolve_moves.loc[resolve_moves['start'] == row['start'], 'end'] = row['start']
						resolve_moves.loc[resolve_moves['start'] == row['start'], 'count'] = 0
						submitted_moves_rerun = self.check_successful_order(units_df, territories_df, pd.concat([rest, resolve_moves]).sort_index(), row['start'], row['start'], success, self_loops)
						try:
							resolve_moves.loc[resolve_moves['start'] == row['start']] = submitted_moves_rerun[submitted_moves_rerun['start'] == row['start']].values
						except ValueError as e:
							resolve_moves.loc[resolve_moves['start'] == row['start']] = submitted_moves_rerun[submitted_moves_rerun['start'] == row['start']].values[0]
				submitted_moves_df = pd.concat([rest, resolve_moves]).sort_index()
		elif not success:
			resolve_moves = submitted_moves_df[submitted_moves_df['end'].isin([end, end_coast])]
			rest = submitted_moves_df.drop(resolve_moves.index)
			resolve_moves_diff_owner = resolve_moves[(resolve_moves['start']==start)|(resolve_moves['owner']!=resolve_moves.loc[resolve_moves['start']==start, 'owner'].values[0])] #remove same owners since can't dislodge self
			maximum = resolve_moves_diff_owner['count'].max()
			if resolve_moves_diff_owner.loc[resolve_moves_diff_owner['start'] == start, 'count'].values[0] != maximum:
				resolve_moves.loc[resolve_moves['start'] == start, 'dislodged'] = True
				resolve_moves.loc[resolve_moves['start'] == start, 'order'] = 'retreat'
			else:
				resolve_moves.loc[resolve_moves['start'] == start, 'dislodged'] = False
			submitted_moves_df = pd.concat([rest, resolve_moves]).sort_index()
		calc_time_diff(begin, 'check_successful_order')
		return submitted_moves_df

	def make_submitted_moves_df(self, units_df, territories_df, orders_list):
		begin = time()
		submitted_moves = {'start': [], 'order': [], 'end': [], 'support': [], 'convoy': []}
		for move in orders_list:
			if len(move) == 3:
				if move[1] in ['move', 'retreat', 'hold']:
					start, order, end, support, convoy = self.submit_move(units_df, territories_df, move[0], move[1], move[2])
				elif move[1] == 'support':
					start, order, end, support, convoy = self.submit_move(units_df, territories_df, move[0], move[1], support = move[2])
				elif move[1] == 'convoy':
					start, order, end, support, convoy = self.submit_move(units_df, territories_df, move[0], move[1], convoy = move[2])
				elif move[1] == 'disband':
					start, order, end, support, convoy = *move, None, None
			elif len(move) == 2:
				start, order, end, support, convoy = self.submit_move(units_df, territories_df, move[0], move[1])
			else:
				start, order, end, support, convoy = self.submit_move(units_df, territories_df, move[0])
			try:
				submitted_moves['start'].append(start)
				submitted_moves['order'].append(order)
				submitted_moves['end'].append(end)
				submitted_moves['support'].append(support)
				submitted_moves['convoy'].append(convoy)
			except UnboundLocalError:
				print(move)
				print(orders_list)
		calc_time_diff(begin, 'make_submitted_moves_df')
		return pd.DataFrame(submitted_moves)

	def resolve_submitted_moves(self, units_df, territories_df, orders_list):
		begin = time()
		original_orders = self.make_submitted_moves_df(units_df, territories_df, orders_list)
		original_orders = original_orders.merge(units_df, how = 'left', left_on = 'start', right_on = 'location').drop('location', axis = 1)
		submitted_moves_df = copy.deepcopy(original_orders)
		support_count = self.count_support(submitted_moves_df)
		submitted_moves_df = submitted_moves_df.merge(support_count[['start', 'count']], on = 'start', how = 'left')# right_on = 'index' .drop('index', axis = 1)
		submitted_moves_df['count'].fillna(0)
		submitted_moves_df['count'] = submitted_moves_df['count'].fillna(0)
		submitted_moves_df['success'] = None
		submitted_moves_df['dislodged'] = False
		self_loops = self.detect_self_loops(submitted_moves_df)
		submitted_moves_df = submitted_moves_df.sort_values('order', ascending = False).reset_index(drop=True) #ensure look at support first
		for ind in range(len(submitted_moves_df)):
			row = submitted_moves_df.loc[ind]
			if row['success'] is None:
				start, end, success = row[['start', 'end', 'success']]
				submitted_moves_df = self.check_successful_order(units_df, territories_df, submitted_moves_df, start, end, success, self_loops)
		calc_time_diff(begin, 'resolve_submitted_moves')
		return submitted_moves_df, original_orders

	def update_territories_df(self, units_df, territories_df, season):
		begin = time()
		old_territories_df = copy.deepcopy(territories_df)
		for ind, row in units_df.iterrows():
			territories_df.loc[territories_df['country'].str.split('_').str[0]==row['location'].split('_')[0], 'controlled_by'] = row['owner']
		if season == 'fall':
			for r in units_df.itertuples(index=None, name=None):
				territories_df.loc[territories_df['country']==r[0], 'sc_control'] = r[2]
		calc_time_diff(begin, 'update_territories_df')
		return old_territories_df, territories_df

	def make_convoy_moves(self, units_df, unit):
		begin = time()
		convoy_sub = [p for p in self.convoy_pairs if p[0] == unit]
		convoy_moves = []
		for c_1, c_2 in convoy_sub:
			G_sub = nx.Graph(self.G.subgraph([x for x,y in self.G.nodes(data=True) if (x in [c_1, c_2]) or (y['unit_type'] == 'fleet' and not y['coast'] and y in units_df['location'].tolist())]))
			if G_sub.has_edge(c_1, c_2):
				G_sub.remove_edge(c_1, c_2)
			if nx.has_path(G_sub, c_1, c_2):
				convoy_moves.append((c_1, 'move', c_2))
		calc_time_diff(begin, 'make_convoy_moves')
		return convoy_moves

	def make_all_possible_orders(self, units_df, territories_df, unit, retreat = False, excluded_ends = [], allies = []):
		begin = time()
		orders_list = []
		if not retreat:
			orders_list.append((unit, 'hold', unit))
			if unit in self.convoyable_countries and units_df.loc[units_df['location']==unit, 'type'].values[0]=='army':
				orders_list.extend(self.make_convoy_moves(units_df, unit))
		connected_countries = [con for con in territories_df.loc[territories_df['country']==unit, 'connected_to'].values[0] if con not in excluded_ends]
		for c in connected_countries:
			orders_list.append((unit, 'move', c))
			if not retreat:
				try:
					orders_list.append((unit, 'support', (c, c)))
					for c_2 in territories_df.loc[territories_df['country']==c, 'connected_to'].values[0]:
						if c_2 != unit:
							orders_list.append((unit, 'support', (c_2, c)))
					if territories_df[territories_df['country']==unit]['unit_type'].values[0] == 'fleet' and not territories_df[territories_df['country']==unit]['coast'].values[0]:
						convoyable_sub = [p for p in self.convoyable_countries if p[0] in units_df['location'].tolist()]
						for con in convoyable_sub:
							orders_list.append((unit, 'convoy', (con[0], con[1])))
				except Exception as e:
					print(f"{c=}")
					print(territories_df.head())
					print(territories_df.info())
					raise e
		final_possible_orders = []
		for o in orders_list:
			try:
				if o[1] == 'support' and o[2][0] in units_df['location'].tolist() and units_df[units_df['location']==o[2][0]]['owner'].values[0] in allies:
					self.check_order(units_df, territories_df, o[0], o[1], support = o[2])
				elif o[1] == 'convoy' and o[2][0] in units_df['location'].tolist() and units_df[units_df['location']==o[2][0]]['owner'].values[0] in allies:
					self.check_order(units_df, territories_df, o[0], o[1], convoy = o[2])
				else:
					self.check_order(units_df, territories_df, o[0], o[1], o[2])
				final_possible_orders.append(o)
			except AssertionError:
				continue
		calc_time_diff(begin, 'make_all_possible_orders')
		return final_possible_orders

	def make_orders(self, units_df, territories_df, policy, loss_fn, original_orders = None, model = None, sea_model = None, land_model = None, allies = {}, naive_policy = None, active_country = None):

		begin = time()

		make_orders_timing_dict = {}
		if model is None:
			assert sea_model is not None and land_model is not None, 'land_model arg is {}, sea_model arg is {}, both need to have models'.format(land_model, sea_model)
			sea_policy_country_list = territories_df[territories_df['unit_type'].isin(['fleet', 'both'])]['country'].tolist()
			land_policy_country_list = territories_df[territories_df['unit_type'].isin(['army', 'both'])]['country'].tolist()
		else:
			policy_country_list = territories_df['country'].tolist()
		make_orders_timing_dict['checking_models'] = time() - begin
		if original_orders is not None:
			excluded_ends = list(set(original_orders['start'].tolist() + original_orders['end'].tolist())) #account for bounces
			excluded_ends = [c.split('_')[0] for c in excluded_ends]
			excluded_ends = territories_df.loc[territories_df['country'].str.split('_').str[0].isin(excluded_ends), 'country'].tolist() #account for coast name differences
			retreat = True
		else:
			excluded_ends = []
			retreat = False
		make_orders_timing_dict['excluding_ends'] = time() - begin
		# if active_country is None:
		orders_dict = {c:{'target': None, 'out_probs': None, 'grads': None, 'order': None, 'model_ver': None} for c in units_df['location'].tolist()}
		# else:
		# 	orders_dict = {c:{'target': None, 'out_probs': None, 'grads': None, 'order': None, 'model_ver': None} for c in units_df[units_df['owner']==active_country]['location'].tolist()}
		## Need to add logic for making allies list
		make_orders_timing_dict['predict_target_for_units'] = []
		make_orders_timing_dict['make_all_possible_orders_for_units'] = []
		make_orders_timing_dict['find_group_best_move_for_targets'] = []
		if not retreat:
			unit_all_orders_dict = {}
			for c, owner, unit_type in list(units_df[['location', 'owner', 'type']].itertuples(False, None)):
				allies_sub = allies[owner]
				if model is None:
					if unit_type=='fleet':
						# print(active_country)
						# print(owner)
						if active_country is not None and owner == active_country:
							# print("caught")
							target, out_probs, grads = policy(sea_policy_country_list, c, owner, sea_model, units_df, territories_df, loss_fn)
						else:
							target, out_probs, grads = naive_policy(sea_policy_country_list, 64)
						model_ver = 'sea'
					elif unit_type=='army':
						if active_country is not None and owner == active_country:
							# print('caught')
							target, out_probs, grads = policy(land_policy_country_list, c, owner, land_model, units_df, territories_df, loss_fn)
						else:
							target, out_probs, grads = naive_policy(land_policy_country_list, 56)
						model_ver = 'land'
					else:
						raise TypeError('Unknown unit type: {}'.format(unit_type))
					# print(grads)
					# print(out_probs)
					make_orders_timing_dict['predict_target_for_units'].append(time() - begin)
				unit_all_orders_dict[c] = {'orders': self.make_all_possible_orders(units_df, territories_df, c, allies = allies_sub), 'target': target, 'owner': owner, 'allies': allies_sub}
				make_orders_timing_dict['make_all_possible_orders_for_units'].append(time() - begin)
				# print('\n\n\n+++++++++++++')
				# print(grads)
				# print('\n\n\n+++++++++++++')
				orders_dict[c]['target'] = target
				orders_dict[c]['out_probs'] = out_probs
				orders_dict[c]['grads'] = grads
				orders_dict[c]['model_ver'] = model_ver
				orders_dict[c]['active_country'] = owner == active_country
			# print(orders_dict)
			group_moves_dict = {}
			for d in set([v['target'] for v in unit_all_orders_dict.values()]):
				group_moves_dict[d] = {k: {'orders':v['orders'], 'owner':v['owner'], 'allies':v['allies']} for k, v in unit_all_orders_dict.items() if v['target']==d}
			for target, sub_moves_dict in group_moves_dict.items():
				orders_dict = self.find_group_best_move(units_df, territories_df, target, allies, sub_moves_dict, orders_dict)
				make_orders_timing_dict['find_group_best_move_for_targets'].append(time() - begin)
			# print(orders_dict)
			orders_list = []
			grads_list = []
			out_probs_list = []
			targets_list = []
			model_ver_list = []
			active_country_list = []
			for k, v in orders_dict.items():
				orders_list.append(v['order'])
				grads_list.append(v['grads'])
				out_probs_list.append(v['out_probs'])
				targets_list.append(v['target'])
				model_ver_list.append(v['model_ver'])
				active_country_list.append(v['active_country'])
			# print(grads_list)
		else:
			orders_list = []
			grads_list = []
			out_probs_list = []
			targets_list = []
			model_ver_list = []
			active_country_list = []
			for c in units_df['location'].tolist():
				final_possible_orders = self.make_all_possible_orders(units_df, territories_df, c, retreat, excluded_ends)
				try:
					order, out_probs, grads = random.choice(final_possible_orders), np.nan, np.nan
				except IndexError:
					order, out_probs, grads = (c, 'disband', c), np.nan, np.nan
				orders_list.append(order)
				grads_list.append(grads)
				out_probs_list.append(out_probs)
				targets_list.append(None)
				model_ver_list.append(None)
				active_country_list.append(None)
			make_orders_timing_dict['retreat'] = time() - begin
			make_orders_timing_dict_list.append(make_orders_timing_dict)
		calc_time_diff(begin, 'make_orders')
		return orders_list, out_probs_list, grads_list, targets_list, model_ver_list, active_country_list

	def find_group_best_move(self, units_df, territories_df, target, allies, sub_moves_dict, final_orders_dict):
		begin = time()
		target_clean = target.split('_')[0] #allow for coast naming differences
		G_sea_sub = nx.Graph(self.G.subgraph([x for x,y in self.G.nodes(data=True) if y['unit_type'] in ['fleet', 'both']]))
		drop_edges =[]
		for n_1, n_2, d, in G_sea_sub.edges(data=True):
			sub = territories_df[territories_df['country'].isin([n_1, n_2])]
			if sum(sub['unit_type']=='both')==2 and not \
				sub[(sub['country'].isin(set([a for b in sub['connected_to'].tolist() for a in b])))&(set['unit_type']=='fleet')].shape[0] > 0:
			# if territories_df.loc[territories_df['country']==n_1, 'unit_type'].values[0] == 'both' and \
			#     territories_df.loc[territories_df['country']==n_2, 'unit_type'].values[0] == 'both' and not \
			#     any([t == 'fleet' for t in territories_df.loc[territories_df['country'].isin(set(territories_df.loc[territories_df['country']==n_1, 'connected_to'].values[0])&set(territories_df.loc[territories_df['country']==n_2, 'connected_to'].values[0])), 'unit_type'].tolist()]):
					drop_edges.append((n_1, n_2))
		
		for n_1, n_2 in drop_edges:
			G_sea_sub.remove_edge(n_1, n_2)
		G_land_sub = nx.Graph(self.G.subgraph([x for x,y in self.G.nodes(data=True) if y['unit_type'] in ['army', 'both'] and not y['coast']])) # or (y['unit_type']=='fleet' and x in units_df['location'].tolist())
		for c, v in sub_moves_dict.items():
			adj_control = [units_df[units_df['location']==l]['owner'].values for l in territories_df[territories_df['country']==c]['connected_to'].values[0]+[c]]
			adj_control = [a[0] if len(a)>0 else None for a in adj_control]
			# add in counts for convoys
			adj_count = Counter(adj_control)
			sub_moves_dict[c]['adj_count'] = adj_count #[target]
			sub_moves_dict[c]['adj_ratio'] = (len(adj_control)-(adj_count[None]+sum([adj_count[a] for a in v['allies']])))/sum([adj_count[a] for a in v['allies']]) #[target]
			sub_moves_dict[c]['num_choices'] = len(v['orders']) #[target]
			sub_moves_dict[c]['num_cut_sup'] = units_df[(units_df['location'].isin(set(territories_df[territories_df['country']==c]['connected_to'].values[0])-set(territories_df[territories_df['country']==target]['connected_to'].values[0])))&(~units_df['owner'].isin(v['allies']))].shape[0] #[target]
			try:
				if units_df[units_df['location']==c]['type'].values[0] == 'fleet':
					sub_moves_dict[c]['dist_target'] = nx.shortest_path_length(G_sea_sub, c, target) #[target]
				else:
					sub_moves_dict[c]['dist_target'] = nx.shortest_path_length(G_land_sub, c, target) #[target]
			except:# Exception as e
				sub_moves_dict[c]['dist_target'] = 1000
				# nx.draw(G_sea_sub)
				# nx.draw(G_land_sub)
				# raise e
			#print(c)
			#print(sub_moves_dict[c]['dist_target'])
			#number of ways support can be cut by countries not adjacent to target
		# top_choices = [k for k, v in sub_moves_dict.items() if v['dist_target'] <= 1]
		for owner in set([v['owner'] for v in sub_moves_dict.values()]):
			sub_moves_allies = {k:v for k, v in sub_moves_dict.items() if v['owner'] in allies[owner]}
			top_choices = [k for k, v in sub_moves_allies.items() if v['dist_target'] <= 1]
			if len(top_choices) > 1:
				top_choices = [c for c in top_choices if sub_moves_allies[c]['adj_ratio']==min([vals['adj_ratio'] for vals in sub_moves_allies.values()])]
			#print(top_choices)
			if len(top_choices) > 1:
				top_choices = [c for c in top_choices if sub_moves_allies[c]['num_cut_sup']==max([vals['num_cut_sup'] for vals in sub_moves_allies.values()])]
			#print(top_choices)
			if len(top_choices) > 1:
				for c in top_choices:
					inv_imp = 0
					sc, home_sc, sc_control = territories_df[territories_df['country']==c][['sc', 'start_control', 'sc_control']].iloc[0].tolist()
					current_occupy = units_df[units_df['location']==c]['owner'].values[0]
					if sc:
						inv_imp += 1
						if sc_control != current_occupy:
							inv_imp += 1
						elif home_sc != current_occupy and sub_moves_dict[c]['adj_ratio'] == 0:
							inv_imp += 1
					sub_moves_allies[c]['sc_calc'] = inv_imp
				top_choices = [c for c in top_choices if sub_moves_allies[c]['sc_calc'] == min([sub_moves_allies[c]['sc_calc'] for c in top_choices])]
			#print(top_choices)
			if len(top_choices) > 1:
				top_choice = random.choice(top_choices)
			elif len(top_choices) == 1:
				top_choice = top_choices[0]
			else:
				top_choice = None
			if top_choice is not None:
				top_choice_clean = top_choice.split('_')[0] #allow for coast naming differences
			# final_orders = {}
			# print(top_choice)
			for k, v in sub_moves_allies.items():
				# print(k)
				#print(v)
				k_clean = k.split('_')[0] #allow for coast naming differences
				if top_choice is None: # this is mainly a bandaid, figure out a better way to deal with top_choice==None or make it not possible
					# print('a')
					if v['dist_target'] == 0:
						order = (k, 'hold', target)
					elif v['dist_target'] == 1:
						order = (k, 'move', target)
					else:
						con_dist = {}
						for con in territories_df[territories_df['country']==k]['connected_to'].values[0]:
							if units_df[units_df['location']==k]['type'].values[0] == 'fleet' and\
								territories_df[territories_df['country']==con]['unit_type'].values[0] in ['fleet', 'both'] and\
								nx.shortest_path_length(G_sea_sub, k, con) == 1:
								try:
									con_dist[con] = nx.shortest_path_length(self.G, con, target)
								except:
									con_dist[con] = 1000 #just arbitrarily large number
							elif units_df[units_df['location']==k]['type'].values[0] == 'army' and territories_df[territories_df['country']==con]['unit_type'].values[0] in ['army', 'both']:
								try:
									con_dist[con] = nx.shortest_path_length(self.G, con, target)
								except:
									con_dist[con] = 1000 #just arbitrarily large number
							else:
								con_dist[con] = 1000 #just arbitrarily large number
						order = (k, 'move', random.choice([ke for ke, va in con_dist.items() if va == min(con_dist.values())]))
				elif k == top_choice and v['dist_target'] == 0:
					# print('b')
					order = (k, 'hold', target)
				elif k == top_choice and v['dist_target'] == 1:
					# print('c')
					if '_' in target:
						for t in [tar for tar in territories_df[territories_df['country']==k]['connected_to'].values[0] if target_clean in tar]:
							try:
								self.check_order(units_df, territories_df, k, 'move', t)
								break
							except AssertionError:
								continue
						order = (k, 'move', t)
					else:
						order = (k, 'move', target)
				elif k != top_choice and target is not None and v['dist_target'] == 1:
					# print('d')
					order = (k, 'support', (top_choice, target))
				elif (k, 'convoy', (top_choice, target)) in v['orders']:
					# print('e')
					# if part of a convoy, choose convoy
					order = (k, 'convoy', (top_choice, target))
				elif top_choice in sub_moves_allies.keys() and (k, 'support', (top_choice, top_choice)) in v['orders']:
					# print('f')
					order = (k, 'support', (top_choice, top_choice))
				elif v['dist_target'] > 1:
					# print('g')
					#print('non-adjacent caught')
					con_dist = {}
					#print(units_df[units_df['location']==k]['type'].values[0])
					for con in territories_df[territories_df['country']==k]['connected_to'].values[0]:
						#print(con, target)
						#print(territories_df[territories_df['country']==con])
						if units_df[units_df['location']==k]['type'].values[0] == 'fleet' and\
							territories_df[territories_df['country']==con]['unit_type'].values[0] in ['fleet', 'both'] and\
							nx.shortest_path_length(G_sea_sub, k, con) == 1:
							#print('fleet caught')
							try:
								con_dist[con] = nx.shortest_path_length(self.G, con, target)
							except:
								#print('fleet except caught')
								con_dist[con] = 1000 #just arbitrarily large number
						elif units_df[units_df['location']==k]['type'].values[0] == 'army' and territories_df[territories_df['country']==con]['unit_type'].values[0] in ['army', 'both']:
							#print('army caught')
							try:
								con_dist[con] = nx.shortest_path_length(self.G, con, target)
							except:
								#print('army except caught')
								con_dist[con] = 1000 #just arbitrarily large number
						else:
							con_dist[con] = 1000 #just arbitrarily large number
					# move to connected territory closest to target
					#print(con_dist)
					order = (k, 'move', random.choice([ke for ke, va in con_dist.items() if va == min(con_dist.values())]))
				# elif territories_df[territories_df['country']==k]['unit_type'].values[0]=='fleet' and not top_choice in territories_df[territories_df['country']==target]['connected_to'].values[0]:
				#     # if part of a convoy, choose convoy
				#     order = (k, 'convoy', (top_choice, target))
				else:
					print('find_group_best_move last else')
					order = (k, 'hold', k)
				assert order in v['orders'], 'Selected order not in possible order list, please check. Selected order: {}, Target: {}, Top Choice: {}, Possible orders: {}'.format(order, target, top_choice, sub_moves_allies)
				# final_orders[k] = order
				final_orders_dict[k]['order'] = order
		calc_time_diff(begin, 'find_group_best_move')
		return final_orders_dict