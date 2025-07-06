from time import time
import networkx as nx
import pandas as pd
import numpy as np
from uuid import uuid4
import json
import pickle
import os
import logging
import traceback

logging.basicConfig(filename = "utils_log.txt")
logger = logging.getLogger("game_assets")

def set_up_game():
    begin = time()
    territories = {
        'Ankara':['Smyrna', 'Constantinople', 'Black Sea', 'Armenia'],
        'Belgium': ['English Channel', 'North Sea', 'Picardy', 'Burgundy', 'Ruhr', 'Holland'],
        'Berlin': ['Baltic Sea', 'Kiel', 'Munich', 'Silesia', 'Prussia'],
        'Brest': ['English Channel', 'Mid Atlantic Ocean', 'Picardy', 'Paris', 'Gascony'],
        'Budapest': ['Galicia', 'Vienna', 'Trieste', 'Serbia', 'Rumania'],
        'Bulgaria': ['Rumania', 'Black Sea', 'Constantinople', 'Aegean Sea', 'Greece', 'Serbia'],
        'Bulgaria_ec': ['Rumania', 'Black Sea', 'Constantinople'],
        'Bulgaria_sc': ['Greece', 'Aegean Sea', 'Constantinople'],
        'Constantinople': ['Black Sea', 'Ankara', 'Smyrna', 'Aegean Sea', 'Bulgaria', 'Bulgaria_ec', 'Bulgaria_sc'],
        'Denmark': ['Helgoland Bight', 'North Sea', 'Skagerrack', 'Sweden', 'Baltic Sea', 'Kiel'],
        'Edinburgh': ['Norwegian Sea', 'North Sea', 'Yorkshire', 'Liverpool', 'Clyde'],
        'Greece': ['Aegean Sea', 'Ionian Sea', 'Albania', 'Serbia', 'Bulgaria', 'Bulgaria_sc'],
        'Holland': ['Helgoland Bight', 'North Sea', 'Belgium', 'Ruhr', 'Kiel'],
        'Kiel': ['Ruhr', 'Holland', 'Helgoland Bight', 'Denmark', 'Baltic Sea', 'Berlin', 'Munich'],
        'Liverpool': ['Clyde', 'Edinburgh', 'Yorkshire', 'Whales', 'North Atlantic Ocean', 'Irish Sea'],
        'London': ['North Sea', 'English Channel', 'Yorkshire', 'Whales'],
        'Marseiles': ['Burgundy', 'Gascony', 'Spain', 'Spain_sc', 'Piedmont', 'Gulf of Lyon'],
        'Moscow': ['St Petersburg', 'Livonia', 'Warsaw', 'Ukraine', 'Sevastopol'],
        'Munich': ['Berlin', 'Kiel', 'Ruhr', 'Burgundy', 'Tyrolia', 'Bohemia', 'Silesia'],
        'Naples': ['Ionian Sea', 'Tyrrhenian Sea', 'Rome', 'Apulia'],
        'Norway': ['Norwegian Sea', 'Barents Sea', 'St Petersburg', 'St Petersburg_nc', 'North Sea', 'Skagerrack', 'Sweden', 'Finland'],
        'Paris': ['Picardy', 'Brest', 'Gascony', 'Burgundy'],
        'Portugal': ['Spain', 'Spain_nc', 'Spain_sc', 'Mid Atlantic Ocean'],
        'Rome': ['Tuscany', 'Venice', 'Apulia', 'Naples', 'Tyrrhenian Sea'],
        'Rumania': ['Sevastopol', 'Ukraine', 'Galicia', 'Budapest', 'Serbia', 'Bulgaria', 'Bulgaria_ec', 'Black Sea'],
        'St Petersburg': ['Moscow', 'Livonia', 'Gulf of Bothnia', 'Finland', 'Norway', 'Barents Sea'],
        'St Petersburg_nc': ['Barents Sea', 'Norway'],
        'St Petersburg_sc': ['Finland', 'Gulf of Bothnia', 'Livonia'],
        'Serbia': ['Budapest', 'Trieste', 'Albania', 'Greece', 'Bulgaria', 'Rumania'],
        'Sevastopol': ['Armenia', 'Black Sea', 'Rumania', 'Ukraine', 'Moscow'],
        'Smyrna': ['Syria', 'Eastern Mediterranean', 'Aegean Sea', 'Constantinople', 'Armenia', 'Ankara'],
        'Spain': ['Marseiles', 'Gulf of Lyon', 'Western Mediterranean', 'Mid Atlantic Ocean', 'Gascony', 'Portugal'],
        'Spain_nc': ['Mid Atlantic Ocean', 'Gascony', 'Portugal'],
        'Spain_sc': ['Portugal', 'Mid Atlantic Ocean', 'Western Mediterranean', 'Gulf of Lyon', 'Marseiles'],
        'Sweden': ['Norway', 'Finland', 'Gulf of Bothnia', 'Baltic Sea', 'Denmark', 'Skagerrack'],
        'Trieste': ['Venice', 'Tyrolia', 'Vienna', 'Budapest', 'Serbia', 'Albania', 'Adriatic Sea'],
        'Tunis': ['Ionian Sea', 'Tyrrhenian Sea', 'Western Mediterranean', 'North Africa'],
        'Venice': ['Piedmont', 'Tyrolia', 'Trieste', 'Adriatic Sea', 'Apulia', 'Rome', 'Tuscany'],
        'Vienna': ['Galicia', 'Budapest', 'Trieste', 'Tyrolia', 'Bohemia'],
        'Warsaw': ['Livonia', 'Moscow', 'Ukraine', 'Galicia', 'Silesia', 'Prussia'],
        'Clyde': ['Norwegian Sea', 'North Atlantic Ocean', 'Edinburgh', 'Liverpool'],
        'Yorkshire': ['Edinburgh', 'Liverpool', 'Whales', 'London', 'North Sea'],
        'North Atlantic Ocean': ['Norwegian Sea', 'Irish Sea', 'Mid Atlantic Ocean', 'Clyde', 'Liverpool'],
        'Mid Atlantic Ocean': ['North Atlantic Ocean', 'Irish Sea', 'English Channel', 'Western Mediterranean', 'Spain', 'Spain_nc', 'Portugal', 'Spain_sc', 'Gascony', 'Brest', 'North Africa'],
        'Norwegian Sea': ['Barents Sea', 'North Atlantic Ocean', 'North Sea', 'Clyde', 'Edinburgh', 'Norway'],
        'North Sea': ['Norwegian Sea', 'Skagerrack', 'Helgoland Bight', 'English Channel', 'Norway', 'Denmark', 'Holland', 'Belgium', 'London', 'Yorkshire', 'Edinburgh'],
        'English Channel': ['North Sea', 'Belgium', 'Picardy', 'Brest', 'Mid Atlantic Ocean', 'Irish Sea', 'Whales', 'London'],
        'Irish Sea': ['North Atlantic Ocean', 'Liverpool', 'Whales', 'English Channel', 'Mid Atlantic Ocean'],
        'Skagerrack': ['North Sea', 'Norway', 'Sweden', 'Denmark'],
        'Baltic Sea': ['Gulf of Bothnia', 'Livonia', 'Prussia', 'Berlin', 'Kiel', 'Denmark', 'Sweden'],
        'Gulf of Bothnia': ['Sweden', 'Finland', 'St Petersburg', 'St Petersburg_sc', 'Livonia', 'Baltic Sea'],
        'Barents Sea': ['Norwegian Sea', 'Norway', 'St Petersburg', 'St Petersburg_nc'],
        'Western Mediterranean': ['Mid Atlantic Ocean', 'North Africa', 'Tunis', 'Tyrrhenian Sea', 'Gulf of Lyon', 'Spain', 'Spain_sc'],
        'Gulf of Lyon': ['Marseiles', 'Piedmont', 'Tuscany', 'Tyrrhenian Sea', 'Western Mediterranean', 'Spain', 'Spain_sc'],
        'Tyrrhenian Sea': ['Ionian Sea', 'Naples', 'Rome', 'Tuscany', 'Gulf of Lyon', 'Western Mediterranean', 'Tunis'],
        'Ionian Sea': ['Tunis', 'Tyrrhenian Sea', 'Naples', 'Apulia', 'Adriatic Sea', 'Albania', 'Greece', 'Aegean Sea', 'Eastern Mediterranean'],
        'Adriatic Sea': ['Apulia', 'Venice', 'Trieste', 'Albania', 'Ionian Sea'],
        'Aegean Sea': ['Ionian Sea', 'Greece', 'Bulgaria', 'Bulgaria_sc', 'Constantinople', 'Smyrna', 'Eastern Mediterranean'],
        'Eastern Mediterranean': ['Syria', 'Smyrna', 'Aegean Sea', 'Ionian Sea'],
        'Black Sea': ['Sevastopol', 'Armenia', 'Ankara', 'Constantinople', 'Bulgaria', 'Bulgaria_ec', 'Rumania'],
        'Picardy': ['Brest', 'Paris', 'Burgundy', 'Belgium', 'English Channel'],
        'Gascony': ['Brest', 'Paris', 'Burgundy', 'Marseiles', 'Spain', 'Mid Atlantic Ocean', 'Spain_nc'],
        'Burgundy': ['Paris', 'Picardy', 'Belgium', 'Ruhr', 'Munich', 'Marseiles', 'Gascony'],
        'North Africa': ['Tunis', 'Western Mediterranean', 'Mid Atlantic Ocean'],
        'Ruhr': ['Belgium', 'Holland', 'Kiel', 'Munich', 'Burgundy'],
        'Prussia': ['Baltic Sea', 'Livonia', 'Warsaw', 'Silesia', 'Berlin'],
        'Silesia': ['Prussia', 'Warsaw', 'Galicia', 'Bohemia', 'Munich', 'Berlin'],
        'Piedmont': ['Tyrolia', 'Venice', 'Tuscany', 'Marseiles', 'Gulf of Lyon'],
        'Tuscany': ['Venice', 'Rome', 'Tyrrhenian Sea', 'Gulf of Lyon', 'Piedmont'],
        'Apulia': ['Adriatic Sea', 'Venice', 'Ionian Sea', 'Naples', 'Rome'],
        'Tyrolia': ['Munich', 'Bohemia', 'Vienna', 'Trieste', 'Venice', 'Piedmont'],
        'Galicia': ['Warsaw', 'Ukraine', 'Rumania', 'Budapest', 'Vienna', 'Bohemia', 'Silesia'],
        'Bohemia': ['Silesia', 'Galicia', 'Vienna', 'Tyrolia', 'Munich'],
        'Finland': ['Norway', 'St Petersburg', 'St Petersburg_sc', 'Gulf of Bothnia', 'Sweden'],
        'Livonia': ['Gulf of Bothnia', 'St Petersburg', 'St Petersburg_sc', 'Moscow', 'Prussia', 'Baltic Sea', 'Warsaw'],
        'Ukraine': ['Moscow', 'Sevastopol', 'Rumania', 'Galicia', 'Warsaw'],
        'Albania': ['Adriatic Sea', 'Trieste', 'Serbia', 'Greece', 'Ionian Sea'],
        'Armenia': ['Sevastopol', 'Syria', 'Smyrna', 'Ankara', 'Black Sea'],
        'Syria': ['Armenia', 'Eastern Mediterranean', 'Smyrna'],
        'Helgoland Bight': ['Denmark', 'Kiel', 'Holland', 'North Sea'],
        'Whales': ['English Channel', 'Irish Sea', 'Liverpool', 'Yorkshire', 'London']
    }
    
    sea_territories = ['North Atlantic Ocean',
    'Mid Atlantic Ocean',
    'Norwegian Sea',
    'North Sea',
    'English Channel',
    'Irish Sea',
    'Helgoland Bight',
    'Skagerrack',
    'Baltic Sea',
    'Gulf of Bothnia',
    'Barents Sea',
    'Western Mediterranean',
    'Gulf of Lyon',
    'Tyrrhenian Sea',
    'Ionian Sea',
    'Adriatic Sea',
    'Aegean Sea',
    'Eastern Mediterranean',
    'Black Sea',
    'Bulgaria_ec',
    'Bulgaria_sc',
    'St Petersburg_nc',
    'St Petersburg_sc',
    'Spain_nc',
    'Spain_sc']

    coast = ['Bulgaria_ec',
             'Bulgaria_sc',
             'St Petersburg_nc',
             'St Petersburg_sc',
             'Spain_nc',
             'Spain_sc']

    coastal = ['Syria',
               'Smyrna',
               'Constantinople',
               'Rumania',
               'Sevastopol',
               'Armenia',
               'Ankara',
               'Greece',
               'Albania',
               'Trieste',
               'Venice',
               'Apulia',
               'Naples',
               'Rome',
               'Tuscany',
               'Piedmont',
               'Marseiles',
               'Portugal',
               'North Africa',
               'Tunis',
               'Gascony',
               'Brest',
               'Picardy',
               'Belgium',
               'Holland',
               'Kiel',
               'Denmark',
               'Berlin',
               'Prussia',
               'Livonia',
               'Finland',
               'Sweden',
               'Norway',
               'Edinburgh',
               'Yorkshire',
               'London',
               'Whales',
               'Liverpool',
               'Clyde']

    sc = [
        'Ankara',
        'Belgium',
        'Berlin',
        'Brest',
        'Budapest',
        'Bulgaria',
        'Bulgaria_ec',
        'Bulgaria_sc',
        'Constantinople',
        'Denmark',
        'Edinburgh',
        'Greece',
        'Holland',
        'Kiel',
        'Liverpool',
        'London',
        'Marseiles',
        'Moscow',
        'Munich',
        'Naples',
        'Norway',
        'Paris',
        'Portugal',
        'Rome',
        'Rumania',
        'St Petersburg',
        'St Petersburg_nc',
        'St Petersburg_sc',
        'Serbia',
        'Sevastopol',
        'Smyrna',
        'Spain',
        'Spain_nc',
        'Spain_sc',
        'Sweden',
        'Trieste',
        'Tunis',
        'Venice',
        'Vienna',
        'Warsaw',
    ]

    france = [
        'Gascony',
        'Marseiles',
        'Burgundy',
        'Paris',
        'Picardy',
        'Brest'
    ]

    england = [
        'Clyde',
        'Edinburgh',
        'Yorkshire',
        'London',
        'Whales',
        'Liverpool'
    ]

    germany = [
        'Kiel',
        'Berlin',
        'Prussia',
        'Silesia',
        'Munich',
        'Ruhr'
    ]

    italy = [
        'Piedmont',
        'Venice',
        'Apulia',
        'Naples',
        'Rome',
        'Tuscany'
    ]

    russia = [
        'Finland',
        'St Petersburg',
        'St Petersburg_nc',
        'St Petersburg_sc',
        'Moscow',
        'Sevastopol',
        'Ukraine',
        'Warsaw',
        'Livonia'
    ]

    austria = [
        'Bohemia',
        'Tyrolia',
        'Galicia',
        'Vienna',
        'Budapest',
        'Trieste'
    ]

    turkey = [
        'Constantinople',
        'Smyrna',
        'Ankara',
        'Armenia',
        'Syria'
    ]
    
    for k, v in territories.items():
        territories[k] = {'connected_to': v}
        if k in sea_territories:
            territories[k]['unit_type'] = 'fleet'
        elif k in coastal:
            territories[k]['unit_type'] = 'both'
        else:
            territories[k]['unit_type'] = 'army'
        if k in coast:
            territories[k]['coast'] = True
        else:
            territories[k]['coast'] = False
        if k in sc:
            territories[k]['sc'] = True
        else:
            territories[k]['sc'] = False
        if k in france:
            territories[k]['controlled_by'] = 'france'
        elif k in italy:
            territories[k]['controlled_by'] = 'italy'
        elif k in england:
            territories[k]['controlled_by'] = 'england'
        elif k in russia:
            territories[k]['controlled_by'] = 'russia'
        elif k in germany:
            territories[k]['controlled_by'] = 'germany'
        elif k in austria:
            territories[k]['controlled_by'] = 'austria'
        elif k in turkey:
            territories[k]['controlled_by'] = 'turkey'
        else:
            territories[k]['controlled_by'] = 'neutral'
    
    coords_dict = {
    'north_africa_army' : (240, 925),
    'north_africa_fleet' : (240, 885),
    'western_mediterranean' : (320, 850),
    'tunis_army' : (450, 925),
    'tunis_fleet' : (470, 895),
    'ionian_sea' : (620, 925),
    'aegean_sea' : (780, 900),
    'eastern_mediterranean' : (880, 940),
    'syria_fleet' : (1005, 940),
    'syria_army' : (1080, 900),
    'armenia_army' : (1100, 800),
    'armenia_fleet' : (1060, 750),
    'ankara_army' : (930, 800),
    'ankara_fleet' : (930, 760),
    'smyrna_army' : (930, 860),
    'smyrna_fleet' : (900, 900),
    'constantinople_army' : (850, 825),
    'constantinople_fleet' : (830, 804),
    'tyrrhenian_sea' : (500, 830),
    'gulf_of_lyon' : (390, 760),
    'adriatic_sea' : (600, 760),
    'black_sea' : (930, 720),
    'mid_atlantic_ocean' : (100, 580),
    'north_atlantic_ocean' : (150, 200),
    'norwegian_sea' : (475, 130),
    'barents_sea' : (860, 25),
    'gulf_of_bothnia' : (680, 320),
    'baltic_sea' : (610, 435),
    'skagerrack' : (540, 345),
    'north_sea' : (430, 345),
    'helgoland_bight' : (460, 435),
    'irish_sea' : (235, 465),
    'english_channel' : (290, 520),
    'portugal_fleet' : (95, 750),
    'portugal_army' : (125, 740),
    'spain_army' : (225, 755),
    'spain_nc' : (225, 670),
    'spain_sc' : (255, 825),
    'gascony_fleet' : (290, 650),
    'gascony_army' : (320, 670),
    'brest_army' : (310, 585),
    'brest_fleet' : (260, 555),
    'picardy_fleet' : (350, 533),
    'picardy_army' : (375, 543),
    'paris' : (355, 585),
    'burgundy' : (400, 615),
    'marseiles_army' : (405, 690),
    'marseiles_fleet' : (370, 710),
    'piedmont_fleet' : (465, 705),
    'piedmont_army' : (460, 680),
    'venice_army' : (510, 695),
    'venice_fleet' : (530, 710),
    'tuscany_army' : (515, 735),
    'tuscany_fleet' : (490, 735),
    'rome_army' : (535, 770),
    'rome_fleet' : (535, 790),
    'apulia_fleet' : (590, 780),
    'apulia_army' : (600, 810),
    'naples_army' : (570, 805),
    'naples_fleet' : (590, 860),
    'greece_army' : (700, 840),
    'greece_fleet' : (700, 890),
    'bulgaria_sc' : (775, 805),
    'bulgaria_army' : (775, 755),
    'bulgaria_ec' : (815, 755),
    'rumania_fleet' : (830, 710),
    'rumania_army' : (760, 710),
    'serbia' : (690, 740),
    'albania_army' : (675, 815),
    'albania_fleet' : (660, 790),
    'trieste_army' : (620, 700),
    'trieste_fleet' : (600, 740),
    'budapest' : (690, 650),
    'galicia' : (740, 590),
    'vienna' : (615, 615),
    'tyrolia' : (570, 625),
    'bohemia' : (575, 575),
    'munich' : (500, 580),
    'silesia' : (610, 530),
    'prussia_army' : (620, 485),
    'prussia_fleet' : (640, 450),
    'berlin_fleet' : (570, 455),
    'berlin_army' : (565, 495),
    'kiel_army' : (510, 495),
    'kiel_fleet' : (490, 460),
    'ruhr' : (470, 540),
    'belgium_army' : (420, 535),
    'belgium_fleet' : (395, 505),
    'holland_fleet' : (430, 485),
    'holland_army' : (442, 505),
    'denmark_army' : (515, 405),
    'denmark_fleet' : (540, 420),
    'norway_army' : (540, 280),
    'norway_fleet' : (480, 330),
    'sweden_army' : (595, 330),
    'sweden_fleet' : (650, 310),
    'finland_fleet' : (685, 250),
    'finland_army' : (740, 250),
    'st_petersburg_army' : (890, 250),
    'st_petersburg_nc' : (845, 185),
    'st_petersburg_sc' : (800, 300),
    'moscow' : (890, 420),
    'livonia_army' : (740, 420),
    'livonia_fleet' : (730, 375),
    'warsaw' : (700, 515),
    'ukraine' : (830, 550),
    'sevastopol_army' : (960, 550),
    'sevastopol_fleet' : (960, 620),
    'edinburgh_fleet' : (375, 330),
    'edinburgh_army' : (340, 360),
    'yorkshire_army' : (355, 440),
    'yorkshire_fleet' : (360, 405),
    'london_army' : (350, 480),
    'london_fleet' : (390, 465),
    'whales_army' : (315, 455),
    'whales_fleet' : (280, 480),
    'liverpool_army' : (332, 440),
    'liverpool_fleet' : (332, 415),
    'clyde_army' : (325, 375),
    'clyde_fleet' : (335, 295)
    }
    coords_dict = {k.replace('_', ' '):v for k, v in coords_dict.items()}
    
    for k, v in territories.items():
        if v['unit_type'] == 'army':
            try:
                territories[k]['coords'] = {'army': coords_dict[k.lower()], 'fleet': np.nan}
            except KeyError:
                territories[k]['coords'] = {'army': coords_dict[k.lower()+' army'], 'fleet': np.nan}
        elif v['unit_type'] == 'both':
            territories[k]['coords'] = {'army': coords_dict[k.lower()+' army'],
                                        'fleet': coords_dict[k.lower()+' fleet']}
        else:
            if v['coast']:
                territories[k]['coords'] = {'army': np.nan, 'fleet': coords_dict[k.lower()[:-3]+' '+k[-2:]]}
            else:
                territories[k]['coords'] = {'army': np.nan, 'fleet': coords_dict[k.lower()]}
    
    G = nx.Graph()
    for k, v in territories.items():
        G.add_nodes_from([(k, v)])
    for k, v in territories.items():
        for c in v['connected_to']:
            G.add_edge(k, c)
    
    territories_df = pd.DataFrame(territories).T.reset_index()
    territories_df = territories_df.rename(columns = {'index':'country'})
    territories_df['start_control'] = territories_df['controlled_by']
    territories_df['sc_control'] = territories_df['controlled_by']
    
    convoy_pairs = []
    convoyable_countries = territories_df[(territories_df['unit_type']=='both')|(territories_df['country'].isin(['Bulgaria', 'Spain', 'St Petersburg']))]['country'].tolist()
    for c_1 in convoyable_countries:
        for c_2 in convoyable_countries:
            if c_1 != c_2:
                G_sub = nx.Graph(G.subgraph([x for x,y in G.nodes(data=True) if (x in [c_1, c_2]) or (y['unit_type'] == 'fleet' and not y['coast'])]))
                if G_sub.has_edge(c_1, c_2):
                    G_sub.remove_edge(c_1, c_2)
                if nx.has_path(G_sub, c_1, c_2):
                    convoy_pairs.append((c_1, c_2))
    
    units = {'location':['Marseiles',
                         'Brest',
                         'Paris',
                         'Edinburgh',
                         'Liverpool',
                         'London',
                         'Kiel',
                         'Berlin',
                         'Munich',
                         'Venice',
                         'Rome',
                         'Naples',
                         'Vienna', 
                         'Budapest', 
                         'Trieste',
                         'Constantinople',
                         'Ankara',
                         'Smyrna',
                         'St Petersburg_sc',
                         'Moscow',
                         'Warsaw',
                         'Sevastopol'],
             'type':['army',
                     'fleet',
                     'army',
                     'fleet',
                     'army',
                     'fleet',
                     'fleet',
                     'army',
                     'army',
                     'army',
                     'army',
                     'fleet',
                     'army',
                     'army',
                     'fleet',
                     'army',
                     'fleet',
                     'army',
                     'fleet',
                     'army',
                     'army',
                     'fleet'],
             'owner':['france',
                      'france',
                      'france',
                      'england',
                      'england',
                      'england',
                      'germany',
                      'germany',
                      'germany',
                      'italy',
                      'italy',
                      'italy',
                      'austria',
                      'austria',
                      'austria',
                      'turkey',
                      'turkey',
                      'turkey',
                      'russia',
                      'russia',
                      'russia',
                      'russia']}
    units_df = pd.DataFrame(units)
    units_df['unit_id'] = [str(uuid4()) for _ in range(units_df.shape[0])]
    # calc_time_diff(begin, 'set_up_game')
    return territories, territories_df, units_df, G, convoy_pairs, convoyable_countries

def create_game_assets():

    os.mkdir("game_assets")

	territories, territories_df, units_df, G, convoy_pairs, convoyable_countries = set_up_game()

	with open("game_assets/diplomacy_graph.pkl", "wb") as f:
		pickle.dump(G, f)

	with open("game_assets/territories.pkl", "wb") as f:
		pickle.dump(territories, f)
	
    with open("game_assets/convoy_pairs.pkl", "wb") as f:
		pickle.dump(convoy_pairs, f)
	
    with open("game_assets/convoyable_countries.pkl", "wb") as f:
		pickle.dump(convoyable_countries, f)
	
    territories_df.to_csv("game_assets/territories_df.csv", index = False)
	
    units_df.to_csv("game_assets/units_df.csv", index = False)


if __name__ == "__main__":





