# pylint: skip-file
# Allowed libraries 
import numpy as np
import pandas as pd
import scipy as sp
import scipy.special
import heapq as pq
import matplotlib as mp
import matplotlib.pyplot as plt
import math
from itertools import product, combinations
from collections import OrderedDict as odict
import collections
from graphviz import Digraph, Graph
from tabulate import tabulate
import copy
import sys
import os
import datetime
import sklearn
import ast
import re
import pickle
import json



###################################
# Code stub
#
# The only requirement of this file is that is must contain a function called get_action,
# and that function must take sensor_data as an argument, and return an actions_dict
#

outcomeSpace = {'r1':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r2':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r3':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r4':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r5':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r6':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r7':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r8':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r9':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'r10':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'c1':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'c2':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'c3':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside'],
                'outside':['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside']
}

# Sensor locations
# motion_loc = {
#     'motion_sensor1': 'r1',
#     'motion_sensor2': 'r2',
#     'motion_sensor3': 'r3',
#     'motion_sensor4': 'r4',
#     'motion_sensor5': 'r5',
#     'motion_sensor6': 'r6',
#     'motion_sensor7': 'r7',
#     'motion_sensor8': 'r8',
#     'motion_sensor9': 'r9',
#     'motion_sensor10': 'r10',
# }

# camera_loc = {
#     'camera1': 'r1',
#     'camera2': 'r4',
#     'camera3': 'r8',
#     'camera4': 'c3'
# }

# door_loc = {
#     'door_sensor1': ['r1', 'r2'],
#     'door_sensor2': ['r3', 'c1'],
#     'door_sensor3': ['r4', 'c1'],
#     'door_sensor4': ['r4', 'r5'],
#     'door_sensor5': ['r5', 'r6'],
#     'door_sensor6': ['r9', 'c2'],
#     'door_sensor7': ['r10', 'c2'],
#     'door_sensor8': ['r7', 'r8'],
#     'door_sensor9': ['r7', 'c3'],
#     'door_sensor10': ['r2', 'c3'],
#     'door_sensor11': ['r10', 'c3']
# }

# all_sensors = {
#     'motion_sensor1': ['r1'],
#     'motion_sensor2': ['r2'],
#     'motion_sensor3': ['r3'],
#     'motion_sensor4': ['r4'],
#     'motion_sensor5': ['r5'],
#     'motion_sensor6': ['r6'],
#     'motion_sensor7': ['r7'],
#     'motion_sensor8': ['r8'],
#     'motion_sensor9': ['r9'],
#     'motion_sensor10': ['r10'],
#     'camera1': ['r1'],
#     'camera2': ['r4'],
#     'camera3': ['r8'],
#     'camera4': ['c3'],
#     'door_sensor1': ['r1', 'r2'],
#     'door_sensor2': ['r3', 'c1'],
#     'door_sensor3': ['r4', 'c1'],
#     'door_sensor4': ['r4', 'r5'],
#     'door_sensor5': ['r5', 'r6'],
#     'door_sensor6': ['r9', 'c2'],
#     'door_sensor7': ['r10', 'c2'],
#     'door_sensor8': ['r7', 'r8'],
#     'door_sensor9': ['r7', 'c3'],
#     'door_sensor10': ['r2', 'c3'],
#     'door_sensor11': ['r10', 'c3']
# }

# rob_list = ['robot1', 'robot2']
# params = pd.read_csv(...)

# th is the threshold, every 15 secs if the number of people above the th, then the light turns on
th = 0.4

#state is the number of people in each space
state = [0]*13
state.append(20)
state = np.array(state)

#index_r is the dictionary storing each room and its corresponding index of state, and index_l is
#the dictionary storing each light and its corresponding index of state.
index_r = {}
index_l = {}
start = 0
for key in outcomeSpace.keys():
    index_r[key] = start
    if start < 10:
        index_l['lights'+str(start+1)] = start
    start+=1


params = pd.read_csv('state_matrix_5.csv')
# train_matrix stores five time period of markov transition matrix.
tran_matrix = {}
tran_matrix['s1'] = []
tran_matrix['s2'] = []
tran_matrix['s3'] = []
tran_matrix['s4'] = []
tran_matrix['s5'] = []


for r in params:
    if r[-2:] == 's1':
        tran_matrix['s1'].append(list(params[r]))
    elif r[-2:] == 's2':
        tran_matrix['s2'].append(list(params[r]))
    elif r[-2:] == 's3':
        tran_matrix['s3'].append(list(params[r]))
    elif r[-2:] == 's4':
        tran_matrix['s4'].append(list(params[r]))
    elif r[-2:] == 's5':
        tran_matrix['s5'].append(list(params[r]))
        
for l in tran_matrix.keys():
    tran_matrix[l] = np.array(tran_matrix[l])

def get_action(sensor_data):
    # declare state as a global variable so it can be read and modified within this function
    global state
    global tran_matrix
    global index_r
    global index_l
    global th

    #different time use different transition matrix
    if int(sensor_data['time'].minute) < 5 and int(sensor_data['time'].hour) == 8:
        state = state @ tran_matrix['s1']
    elif int(sensor_data['time'].hour) < 12:
        state = state @ tran_matrix['s2']
    elif int(sensor_data['time'].hour) < 14:
        state = state @ tran_matrix['s3']
    elif int(sensor_data['time'].hour) < 17:
        state = state @ tran_matrix['s4']
    elif int(sensor_data['time'].hour) == 17 and int(sensor_data['time'].minute) < 30:
        state = state @ tran_matrix['s4']
    else:
        state = state @ tran_matrix['s5']

   
    if sensor_data['motion_sensor1'] == 'motion' and state[index_r['r1']] < th:
        state[index_r['r1']] = 0.98
    if sensor_data['motion_sensor1'] == 'no motion' and state[index_r['r1']] > th:
        state[index_r['r1']] = 0.36

    if sensor_data['motion_sensor2'] == 'motion' and state[index_r['r2']] < th:
        state[index_r['r2']] = 0.92
    if sensor_data['motion_sensor2'] == 'no motion' and state[index_r['r2']] > th:
        state[index_r['r2']] = 0.24

    if sensor_data['motion_sensor3'] == 'motion' and state[index_r['r3']] < th:
        state[index_r['r3']] = 0.97
    if sensor_data['motion_sensor3'] == 'no motion' and state[index_r['r3']] > th:
        state[index_r['r3']] = 0.31

    if sensor_data['motion_sensor4'] == 'motion' and state[index_r['r4']] < th:
        state[index_r['r4']] = 0.86
    if sensor_data['motion_sensor4'] == 'no motion' and state[index_r['r4']] > th:
        state[index_r['r4']] = 0.10

    if sensor_data['motion_sensor5'] == 'motion' and state[index_r['r5']] < th:
        state[index_r['r5']] = 0.97
    if sensor_data['motion_sensor5'] == 'no motion' and state[index_r['r5']] > th:
        state[index_r['r5']] = 0.14

    if sensor_data['motion_sensor6'] == 'motion' and state[index_r['r6']] < th:
        state[index_r['r6']] = 0.68
    if sensor_data['motion_sensor6'] == 'no motion' and state[index_r['r6']] > th:
        state[index_r['r6']] = 0.04

    if sensor_data['motion_sensor7'] == 'motion' and state[index_r['r7']] < th:
        state[index_r['r7']] = 0.84
    if sensor_data['motion_sensor7'] == 'no motion' and state[index_r['r7']] > th:
        state[index_r['r7']] = 0.09

    if sensor_data['motion_sensor8'] == 'motion' and state[index_r['r8']] < th:
        state[index_r['r8']] = 0.80
    if sensor_data['motion_sensor8'] == 'no motion' and state[index_r['r8']] > th:
        state[index_r['r8']] = 0.06
    
    if sensor_data['motion_sensor9'] == 'motion' and state[index_r['r9']] < th:
        state[index_r['r9']] = 0.98
    if sensor_data['motion_sensor9'] == 'no motion' and state[index_r['r9']] > th:
        state[index_r['r9']] = 0.15
    
    if sensor_data['motion_sensor10'] == 'motion' and state[index_r['r10']] < th:
        state[index_r['r10']] = 0.95
    if sensor_data['motion_sensor10'] == 'no motion' and state[index_r['r10']] > th:
        state[index_r['r10']] = 0.14

    #Camera
    if sensor_data['camera1'] != 0 and state[index_r['r1']] < th and sensor_data['camera1'] != None:
        state[index_r['r1']] = int(sensor_data['camera1'])

    if sensor_data['camera2'] != 0 and state[index_r['r4']] < th and sensor_data['camera2'] != None:
        state[index_r['r4']] = int(sensor_data['camera2'])
    
    if sensor_data['camera3'] != 0 and state[index_r['r8']] < th and sensor_data['camera3'] != None:
        state[index_r['r3']] = int(sensor_data['camera3'])
    
    if sensor_data['camera4'] != 0 and state[index_r['c3']] < th and sensor_data['camera4'] != None:
        state[index_r['c3']] = int(sensor_data['camera4'])


   #for the door sensors
    if sensor_data['door_sensor1'] and int(sensor_data['door_sensor1']) > 0 and  (state[index_r['r1']] < th or state[index_r['r2']] < th):
        if state[index_r['r1']] < th:
            state[index_r['r1']] = 0.92
        if state[index_r['r2']] < th:
            state[index_r['r2']] = 0.81

    if sensor_data['door_sensor2'] and int(sensor_data['door_sensor2']) > 0 and  (state[index_r['r3']] < th or state[index_r['c1']] < th):
        if state[index_r['r3']] < th:
            state[index_r['r3']] = 0.84
        if state[index_r['c1']] < th:
            state[index_r['c1']] = 0.82

    if sensor_data['door_sensor3'] and int(sensor_data['door_sensor3']) > 0 and  (state[index_r['r4']] < th or state[index_r['c1']] < th):
        if state[index_r['r4']] < th:
            state[index_r['r4']] = 0.68
        if state[index_r['c1']] < th:
            state[index_r['c1']] = 0.81

    if sensor_data['door_sensor4'] and int(sensor_data['door_sensor4']) > 0 and  (state[index_r['r4']] < th or state[index_r['r5']] < th):
        if state[index_r['r4']] < th:
            state[index_r['r4']] = 0.68
        if state[index_r['r5']] < th:
            state[index_r['r5']] = 0.91

    if sensor_data['door_sensor5'] and int(sensor_data['door_sensor5']) > 0 and  (state[index_r['r5']] < th or state[index_r['r6']] < th):
        if state[index_r['r5']] < th:
            state[index_r['r5']] = 0.98
        if state[index_r['r6']] < th:
            state[index_r['r7']] = 0.65

    if sensor_data['door_sensor6'] and int(sensor_data['door_sensor6']) > 0 and  (state[index_r['r9']] < th or state[index_r['c2']] < th):
        if state[index_r['r9']] < th:
            state[index_r['r9']] = 0.91
        if state[index_r['c2']] < th:
            state[index_r['c2']] = 0.64

    if sensor_data['door_sensor7'] and int(sensor_data['door_sensor7']) > 0 and  (state[index_r['r10']] < th or state[index_r['c2']] < th):
        if state[index_r['r10']] < th:
            state[index_r['r10']] = 0.86
        if state[index_r['c2']] < th:
            state[index_r['c2']] = 0.63

    if sensor_data['door_sensor8'] and int(sensor_data['door_sensor8']) > 0 and  (state[index_r['r7']] < th or state[index_r['r8']] < th):
        if state[index_r['r7']] < th:
            state[index_r['r7']] = 0.71
        if state[index_r['r8']] < th:
            state[index_r['r8']] = 0.67
    
    if sensor_data['door_sensor9'] and int(sensor_data['door_sensor9']) > 0 and  (state[index_r['r7']] < th or state[index_r['c3']] < th):
        if state[index_r['r7']] < th:
            state[index_r['r7']] = 0.69
        if state[index_r['c3']] < th:
            state[index_r['c3']] = 0.93

    if sensor_data['door_sensor10'] and int(sensor_data['door_sensor10']) > 0 and  (state[index_r['r2']] < th or state[index_r['c3']] < th):
        if state[index_r['r2']] < th:
            state[index_r['r2']] = 0.81
        if state[index_r['c3']] < th:
            state[index_r['c3']] = 0.88
    
    if sensor_data['door_sensor11'] and int(sensor_data['door_sensor11']) > 0 and  (state[index_r['r10']] < th or state[index_r['c3']] < th):
        if state[index_r['r10']] < th:
            state[index_r['r10']] = 0.87
        if state[index_r['c3']] < th:
            state[index_r['c3']] = 0.93
    
    #robot sensors
    if sensor_data['robot1'] != None:
        ob_ro1 = sensor_data['robot1'].split('\'')[1]
        num_ro1 = sensor_data['robot1'].split(',')[1].split(')')[0]
        state[index_r[ob_ro1]] = int(num_ro1)
    if sensor_data['robot2'] != None:
        ob_ro2 = sensor_data['robot2'].split('\'')[1]
        num_ro2 = sensor_data['robot2'].split(',')[1].split(')')[0]
        state[index_r[ob_ro2]] = int(num_ro2)

    actions_dict = {}
    for key in index_l:
    
        if state[index_l[key]] > th:
            actions_dict[key] = 'on'
        else:
            actions_dict[key] = 'off'

    return actions_dict