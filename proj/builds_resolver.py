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

from utils.aux_funcs import calc_time_diff

class BuildsMaker:
    def __init__(self, territories, G, convoy_pairs, convoyable_countries):
        self.territories = territories
        self.G = G
        self.convoy_pairs = convoy_pairs
        self.convoyable_countries = convoyable_countries

    def make_builds(self, units_df, territories_df, loss_fn, policy = None, human = False, model = None, unit_rewards = None, active_country=None, naive_builds_policy=None, rewards_calculator=None):
        begin = time()
        builds_output_dict = {}
        for c in ['france','italy','england','russia','germany','austria','turkey']:
            if naive_builds_policy is not None and c!=active_country:
                policy = naive_builds_policy
            # else:
            #     policy = policy
            num_builds = sum((territories_df['sc_control']==c)&(territories_df['sc']==True)&(territories_df['coast']==False)) - sum(units_df['owner']==c)
            if num_builds > 0:
                if human:
                    b = input('Enter builds [(location, unit type)]: ')
                    b = ast.literal_eval(b)
                    for l, ut in b:
                        assert l in territories_df[(territories_df['start_control']==c)&(territories_df['sc'])]['country'].tolist(), 'cannot build in {}, not a starting supply center for {}'.format(l, c)
                        assert l not in units_df['location'], 'cannot build in {}, territory is already occupied'.format(l)
                        assert l in territories_df[territories_df['controlled_by']==c]['country'], "cannot build in {}, no longer under {}'s control".format(l, c)
                        build_country_ut = territories_df.loc[territories_df['country']==l, 'unit_type'].values[0]
                        assert ut == build_country_ut or build_country_ut == 'both', 'cannot build a {} in {}, choose the other unit type'.format(ut, c)
                    b = [(a[0], a[1], c, str(uuid4())) for a in b]
                    units_df = pd.concat([units_df, pd.DataFrame(b, columns = units_df.columns)], ignore_index=True)
                else:
                    for build in range(num_builds, 0, -1):
                        build_policy_out = policy(num_builds, c, model, units_df, loss_fn, territories_df)
                        try:
                            build_loc, army_proba, grads = build_policy_out
                        except ValueError:
                            build_loc, army_proba, grads = build_policy_out, np.nan, np.nan
                        if build_loc[0] is not None:
                            build_loc = (build_loc[0], build_loc[1], c, str(uuid4()))
                            builds_output_dict[build_loc[3]] = {'rewards': [], 'grads': grads, 'owner': c}
                            units_df = pd.concat([units_df, pd.DataFrame([build_loc], columns = units_df.columns)], ignore_index=True)
            if num_builds < 0:
                if human:
                    b = input('Enter units to disband (list of countries): ')
                    units_df = units_df[~units_df['location'].isin(b)]
                else:
                    for disband in range(num_builds, 0, 1):
                        build_rewards_list = []
                        disband_policy_out = policy(num_builds, c, model, units_df, loss_fn, territories_df)
                        try:
                            disband_loc, army_proba, grads = disband_policy_out
                        except ValueError:
                            print(disband_policy_out)
                            disband_loc, army_proba, grads = (disband_policy_out, np.nan), np.nan, np.nan
                        try:
                            unit_id = units_df.loc[units_df['location'] == disband_loc[0], 'unit_id'].values[0]
                        except IndexError as e:
                            print(units_df)
                            print(disband_loc)
                            raise e
                        build_rewards_list.append(rewards_calculator.calc_unit_performance_reward(units_df, c, unit_id, unit_rewards))
                        builds_output_dict[unit_id] = {'rewards': build_rewards_list, 'grads': grads, 'owner': c}
                        units_df = units_df[~(units_df['location']==disband_loc[0])]
        calc_time_diff(begin, 'make_builds')
        return units_df.reset_index(drop=True), builds_output_dict