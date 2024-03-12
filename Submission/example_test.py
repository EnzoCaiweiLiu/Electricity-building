# pylint: disable-all
'''
COMP9418 Assignment 2
This file is similar to the file that will be used to test your assignment
It should be used to make sure you code will work during testing
'''
# Make division default to floating-point, saving confusion
from __future__ import division
from __future__ import print_function

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
import re
import ast


from solution import get_action


# simulator code
class Person:    
    def __init__(self, name, office=None):
        self.name = name
        
    def timestep(self, building_simulator):
        pass

class Motion:
    def __init__(self, name, room):
        self.room = room
        self.name = name
    def get_output(self, room_occupancy):
        pass

class Camera:
    def __init__(self, name, room):
        self.room = room
        self.name = name
    def get_output(self, room_occupancy):
        pass

class DoorSensor:
    def __init__(self, name, rooms):
        self.rooms = rooms #pair of rooms
        self.name = name
    def get_output(self, building_simulator):
        pass

class Robot:
    def __init__(self, name, start_room):
        self.name = name
        self.current_location = start_room
    def get_output(self, building_simulator):
        pass        
    def timestep(self, building_simulator):
        pass
    
# part of the code from the building simulator.
class SmartBuildingSimulatorExample:
    def __init__(self):
        self.data = pd.read_csv('data1.csv')   
        # self.data = pd.read_csv('data2.csv')

        self.num_lights = 10
        self.num_people = 20 
        self.start_time = datetime.time(hour=8,minute=0)
        self.end_time = datetime.time(18,0)
        self.time_step = datetime.timedelta(seconds=15) # 15 seconds
        
        self.current_time = self.start_time

        self.current_electricity_price = 2
        self.productivity_cost = 4 
        # cumulative cost so far today
        self.cost = 0

        self.people = [Person(i) for i in range(1,self.num_people+1)]

        self.room_occupancy = dict([(room, 0) for room in ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'c1', 'c2', 'c3', 'outside']])
        self.room_occupancy['outside'] = self.num_people
        
        # current state of lights
        lights = {}
        for i in range(1,self.num_lights+1):
            lights["lights"+str(i)] = "off"

        self.lights = lights
            
        # set up of all sensors
        self.motion_sensors = [Motion('motion_sensor1',['r1']),Motion('motion_sensor2',['r2']),
                               Motion('motion_sensor3',['r3']),Motion('motion_sensor4',['r4']),
                               Motion('motion_sensor5',['r5']),Motion('motion_sensor6',['r6']),
                               Motion('motion_sensor7',['r7']),Motion('motion_sensor8',['r8']),
                               Motion('motion_sensor9',['r9']),Motion('motion_sensor10',['r10'])]
        
        self.door_sensors = [DoorSensor('door_sensor1',('r1','r2')),DoorSensor('door_sensor2',('c1','r3')),
                             DoorSensor('door_sensor3',('c1','r4')),DoorSensor('door_sensor4',('r4','r5')),
                             DoorSensor('door_sensor5',('r5','r6')),DoorSensor('door_sensor6',('r9','c2')),
                             DoorSensor('door_sensor7',('r10','c2')),DoorSensor('door_sensor8',('r8','r7')),
                             DoorSensor('door_sensor9',('r7','c3')),DoorSensor('door_sensor10',('r2','c3')),
                             DoorSensor('door_sensor11',('c3','r10'))]

        self.camera_sensors = [Camera('camera1',['r1']), Camera('camera2',['r4']),Camera('camera3',['r8']),
                               Camera('camera4',['c3'])]
        
        self.robot_sensors = [Robot('robot1','r1'), Robot('robot2','r7')]

        self.curr_step = 0
        
    def timestep(self, actions_dict=None):
        '''
        actions_dict is a dictionary that maps from action strings to either 'on' or 'off'
        '''
        # do actions
        if actions_dict is not None:
            for key in actions_dict:
                self.lights[key] # test that action exists
                self.lights[key] = actions_dict[key]

        # calculate cost (new)
        self.cost += self.cost_of_prev_timestep(self.current_electricity_price)                
                
        # get data for current timestep (this example test uses saved data instead of randomly simulated data)
        current_data = self.data.iloc[self.curr_step]
        
        # move people 
        for room in self.room_occupancy:
            self.room_occupancy[room] = current_data.loc[room]

        # increment time
        self.current_time = (datetime.datetime.combine(datetime.date.today(), self.current_time) + self.time_step).time()

        # calculate cost (depreciated)
        # self.cost += self.cost_of_prev_timestep(self.current_electricity_price)

        # work out sensor data
        sensor_data = {}
        for sensor in self.motion_sensors:
            sensor_data[sensor.name] = current_data[sensor.name]
        for sensor in self.camera_sensors:
            sensor_data[sensor.name] = current_data[sensor.name]
        for robot in self.robot_sensors:
            robot.timestep(self)
            sensor_data[robot.name] = current_data[robot.name]
        for sensor in self.door_sensors:
            sensor_data[sensor.name] = current_data[sensor.name]

        # To make sure your code can handle this case,
        # set one random sensor to None
        broken_sensor = np.random.choice(list(sensor_data.keys())) 
        sensor_data[broken_sensor] = None

        sensor_data['time'] = self.current_time 

        self.curr_step += 1

        return sensor_data

    def cost_of_prev_timestep(self, electricity_price):
        '''
        calculates the cost incurred in the previous 15 seconds
        '''
        cost = 0
        for light, state in self.lights.items():
            room_num = 'r' + (light[6:]) # extract number from string
            if state == 'on':
                cost += self.current_electricity_price
            elif state == 'off':
                cost += self.productivity_cost*self.room_occupancy[room_num]
            else:
                raise Exception("Invalid light state")
        return cost

simulator = SmartBuildingSimulatorExample()

sensor_data = simulator.timestep()
for i in range(len(simulator.data)-1):
    actions_dict = get_action(sensor_data)
    sensor_data = simulator.timestep(actions_dict)

print(f"Total cost for the day: {simulator.cost} cents")
