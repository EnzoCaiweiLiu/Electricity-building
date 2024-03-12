# pylint: skip-file
'''
COMP9418 Assignment 2
This file is the example code to show how the assignment will be tested.

Name: Caiwei Liu    zID: z5391857

Name: Darren Chong  zID: z5311772
'''

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

#threshold
th = 0.4

#State is the number of people in each space
state = [0]*13
#Outside begins with 20 people
state.append(20)
state = np.array(state)

rooms = {}
lights = {}
location = 0
for key in outcomeSpace.keys():
    rooms[key] = location
    if location < 10:
        lights['lights'+str(location+1)] = location
    location += 1



#State Transition Probability
transition_table = pd.read_csv('tran_matrix_probability.csv')

#Transition at different time
transition = {}
transition['time1'] = []
transition['time2'] = []
transition['time3'] = []

#Append transition probability to transition
for col in transition_table:
    if col[-5:] == 'time1':
        transition['time1'].append(list(transition_table[col]))
    elif col[-5:] == 'time2':
        transition['time2'].append(list(transition_table[col]))
    elif col[-5:] == 'time3':
        transition['time3'].append(list(transition_table[col]))

#Turn it into Numpy Array
for key in transition.keys():
    transition[key] = np.array(transition[key])

def get_action(sensor_data):
    global state
    global transition
    global rooms
    global lights
    global th

    #(8:00 - 8:05)
    if int(sensor_data['time'].minute) < 5 and int(sensor_data['time'].hour) == 8:
        state = state @ transition['time1']
    #(8:05 - 17:00)
    elif int(sensor_data['time'].hour) < 17:
        state = state @ transition['time2']
    #(17:00 - 17:30)
    elif int(sensor_data['time'].hour) == 17 and int(sensor_data['time'].minute) < 30:
        state = state @ transition['time2']
    #(17:00 - 18:00)
    else:
        state = state @ transition['time3']
    
    #Motion Sensor
    #If motion use true positive 
    #Otherwise use False Negative
    #True Positive and False Negative got from reliability_checker
    if sensor_data['motion_sensor1'] == 'motion' and state[rooms['r1']] < th:
        state[rooms['r1']] = 0.98
    if sensor_data['motion_sensor1'] == 'no motion' and state[rooms['r1']] > th:
        state[rooms['r1']] = 0.36

    if sensor_data['motion_sensor2'] == 'motion' and state[rooms['r2']] < th:
        state[rooms['r2']] = 0.92
    if sensor_data['motion_sensor2'] == 'no motion' and state[rooms['r2']] > th:
        state[rooms['r2']] = 0.24

    if sensor_data['motion_sensor3'] == 'motion' and state[rooms['r3']] < th:
        state[rooms['r3']] = 0.97
    if sensor_data['motion_sensor3'] == 'no motion' and state[rooms['r3']] > th:
        state[rooms['r3']] = 0.31

    if sensor_data['motion_sensor4'] == 'motion' and state[rooms['r4']] < th:
        state[rooms['r4']] = 0.86
    if sensor_data['motion_sensor4'] == 'no motion' and state[rooms['r4']] > th:
        state[rooms['r4']] = 0.10

    if sensor_data['motion_sensor5'] == 'motion' and state[rooms['r5']] < th:
        state[rooms['r5']] = 0.97
    if sensor_data['motion_sensor5'] == 'no motion' and state[rooms['r5']] > th:
        state[rooms['r5']] = 0.14

    if sensor_data['motion_sensor6'] == 'motion' and state[rooms['r6']] < th:
        state[rooms['r6']] = 0.68
    if sensor_data['motion_sensor6'] == 'no motion' and state[rooms['r6']] > th:
        state[rooms['r6']] = 0.04

    if sensor_data['motion_sensor7'] == 'motion' and state[rooms['r7']] < th:
        state[rooms['r7']] = 0.84
    if sensor_data['motion_sensor7'] == 'no motion' and state[rooms['r7']] > th:
        state[rooms['r7']] = 0.09

    if sensor_data['motion_sensor8'] == 'motion' and state[rooms['r8']] < th:
        state[rooms['r8']] = 0.80
    if sensor_data['motion_sensor8'] == 'no motion' and state[rooms['r8']] > th:
        state[rooms['r8']] = 0.06
    
    if sensor_data['motion_sensor9'] == 'motion' and state[rooms['r9']] < th:
        state[rooms['r9']] = 0.98
    if sensor_data['motion_sensor9'] == 'no motion' and state[rooms['r9']] > th:
        state[rooms['r9']] = 0.15
    
    if sensor_data['motion_sensor10'] == 'motion' and state[rooms['r10']] < th:
        state[rooms['r10']] = 0.95
    if sensor_data['motion_sensor10'] == 'no motion' and state[rooms['r10']] > th:
        state[rooms['r10']] = 0.14

    #Camera
    #If Camera spots someone change state to whatever camera spots
    #100 probability
    #Does not matter if the camera counts correctly because if camera spots someone means theres someone
    #Even 1 will past the threshold
    if sensor_data['camera1'] != 0 and state[rooms['r1']] < th and sensor_data['camera1'] != None:
        state[rooms['r1']] = int(sensor_data['camera1'])

    if sensor_data['camera2'] != 0 and state[rooms['r4']] < th and sensor_data['camera2'] != None:
        state[rooms['r4']] = int(sensor_data['camera2'])
    
    if sensor_data['camera3'] != 0 and state[rooms['r8']] < th and sensor_data['camera3'] != None:
        state[rooms['r3']] = int(sensor_data['camera3'])
    
    if sensor_data['camera4'] != 0 and state[rooms['c3']] < th and sensor_data['camera4'] != None:
        state[rooms['c3']] = int(sensor_data['camera4'])


    #Door sensors
    if sensor_data['door_sensor1'] and int(sensor_data['door_sensor1']) > 0:
        if state[rooms['r1']] < th:
            state[rooms['r1']] = 0.92
        if state[rooms['r2']] < th:
            state[rooms['r2']] = 0.81
    
    if sensor_data['door_sensor1'] and int(sensor_data['door_sensor1']) == 0:
        if state[rooms['r1']] < th:
            state[rooms['r1']] = 0.78
        if state[rooms['r2']] < th:
            state[rooms['r2']] = 0.54

    if sensor_data['door_sensor2'] and int(sensor_data['door_sensor2']) > 0:
        if state[rooms['r3']] < th:
            state[rooms['r3']] = 0.84
        if state[rooms['c1']] < th:
            state[rooms['c1']] = 0.82
    
    if sensor_data['door_sensor2'] and int(sensor_data['door_sensor2']) == 0:
        if state[rooms['r3']] < th:
            state[rooms['r3']] = 0.77
        if state[rooms['c1']] < th:
            state[rooms['c1']] = 0.40

    if sensor_data['door_sensor3'] and int(sensor_data['door_sensor3']) > 0:
        if state[rooms['r4']] < th:
            state[rooms['r4']] = 0.68
        if state[rooms['c1']] < th:
            state[rooms['c1']] = 0.81
    
    if sensor_data['door_sensor3'] and int(sensor_data['door_sensor3']) == 0:
        if state[rooms['c1']] < th:
            state[rooms['c1']] = 0.48

    if sensor_data['door_sensor4'] and int(sensor_data['door_sensor4']) > 0:
        if state[rooms['r4']] < th:
            state[rooms['r4']] = 0.68
        if state[rooms['r5']] < th:
            state[rooms['r5']] = 0.91
    
    if sensor_data['door_sensor4'] and int(sensor_data['door_sensor4']) == 0:
        if state[rooms['r5']] < th:
            state[rooms['r5']] = 0.70

    if sensor_data['door_sensor5'] and int(sensor_data['door_sensor5']) > 0:
        if state[rooms['r5']] < th:
            state[rooms['r5']] = 0.98
        if state[rooms['r6']] < th:
            state[rooms['r6']] = 0.65

    if sensor_data['door_sensor5'] and int(sensor_data['door_sensor5']) == 0:
        if state[rooms['r5']] < th:
            state[rooms['r5']] = 0.71

    if sensor_data['door_sensor6'] and int(sensor_data['door_sensor6']) > 0:
        if state[rooms['r9']] < th:
            state[rooms['r9']] = 0.91
        if state[rooms['c2']] < th:
            state[rooms['c2']] = 0.64
    
    if sensor_data['door_sensor6'] and int(sensor_data['door_sensor6']) == 0:
        if state[rooms['r9']] < th:
            state[rooms['r9']] = 0.60

    if sensor_data['door_sensor7'] and int(sensor_data['door_sensor7']) > 0:
        if state[rooms['r10']] < th:
            state[rooms['r10']] = 0.86
        if state[rooms['c2']] < th:
            state[rooms['c2']] = 0.63
    
    if sensor_data['door_sensor7'] and int(sensor_data['door_sensor7']) == 0:
        if state[rooms['r10']] < th:
            state[rooms['r10']] = 0.57

    if sensor_data['door_sensor8'] and int(sensor_data['door_sensor8']) > 0:
        if state[rooms['r7']] < th:
            state[rooms['r7']] = 0.71
        if state[rooms['r8']] < th:
            state[rooms['r8']] = 0.67
    
    if sensor_data['door_sensor9'] and int(sensor_data['door_sensor9']) > 0:
        if state[rooms['r7']] < th:
            state[rooms['r7']] = 0.69
        if state[rooms['c3']] < th:
            state[rooms['c3']] = 0.93

    if sensor_data['door_sensor9'] and int(sensor_data['door_sensor9']) == 0:
        if state[rooms['c3']] < th:
            state[rooms['c3']] = 0.71

    if sensor_data['door_sensor10'] and int(sensor_data['door_sensor10']) > 0:
        if state[rooms['r2']] < th:
            state[rooms['r2']] = 0.81
        if state[rooms['c3']] < th:
            state[rooms['c3']] = 0.88
    
    if sensor_data['door_sensor10'] and int(sensor_data['door_sensor10']) == 0:
        if state[rooms['r2']] < th:
            state[rooms['r2']] = 0.57
        if state[rooms['c3']] < th:
            state[rooms['c3']] = 0.69
    
    if sensor_data['door_sensor11'] and int(sensor_data['door_sensor11']) > 0:
        if state[rooms['r10']] < th:
            state[rooms['r10']] = 0.87
        if state[rooms['c3']] < th:
            state[rooms['c3']] = 0.93
    
    if sensor_data['door_sensor11'] and int(sensor_data['door_sensor11']) == 0:
        if state[rooms['c3']] < th:
            state[rooms['c3']] = 0.63
    
    #Robot is a hundred precent correct
    if sensor_data['robot1'] != None:
        observation = sensor_data['robot1'].split('\'')[1]
        num_people = sensor_data['robot1'].split(',')[1].split(')')[0]
        state[rooms[observation]] = int(num_people)
    if sensor_data['robot2'] != None:
        observation = sensor_data['robot2'].split('\'')[1]
        num_people = sensor_data['robot2'].split(',')[1].split(')')[0]
        state[rooms[observation]] = int(num_people)

    #Lights turn on or off
    actions_dict = {}
    #Go through all lights
    for light in lights:
        #If the room has a num people above threshold turn it on
        if state[lights[light]] > th:
            actions_dict[light] = 'on'
        else:
            actions_dict[light] = 'off'

    return actions_dict