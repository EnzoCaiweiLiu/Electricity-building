# pylint: skip-file

#Import necessary file
import numpy as np
import pandas as pd

#Get data
data_1 = pd.read_csv('data1.csv')
data_2 = pd.read_csv('data2.csv')

#Combine data for easy iteration
extended_data = pd.concat([data_1, data_2], ignore_index=True)
sorted_data = extended_data.sort_values(by='time')

#Reliability of robots
#How many does the robot get correct
r1_correct = 0
r2_correct = 0

#Go through dataset
for i in range(len(sorted_data)):
    #Get location of robot
    r1_location = sorted_data['robot1'].iloc[i].split('\'')[1]
    r2_location = sorted_data['robot2'].iloc[i].split('\'')[1]
    #Get how many the robot spots
    r1_observation = sorted_data['robot1'].iloc[i].split(',')[1].split(')')[0]
    r2_observation = sorted_data['robot2'].iloc[i].split(',')[1].split(')')[0]
    #Get how many people there is actually
    num_people_r1 = sorted_data[r1_location].iloc[i]
    num_people_r2 = sorted_data[r2_location].iloc[i]
    #If number of people in the room equal 
    #the number of people the robot spots that means robot is correct
    if int(r1_observation) == int(num_people_r1):
        r1_correct += 1
    
    if int(r2_observation) == int(num_people_r2):
        r2_correct += 1

print("Accuracy of robot 1 : " , r1_correct/(len(sorted_data)))
print("Accuracy of robot 2 : " , r2_correct/(len(sorted_data)))

#Reliability of Motion Sensers

def accuracy_motion(sensor, room):
    #We need to know True positive and False Negative so we can appropriately change our data
    true_pos = 0
    false_neg = 0
    #Total amount fo time the sensors is in motion
    motion = 0
    no_motion = 0

    for i in range(len(sorted_data)):
        #What the motion sensor sense
        observation = sorted_data[sensor].iloc[i]
        #Number of people in room
        num_people = sorted_data[room].iloc[i]
        #If in motion 
        if observation == 'motion':
            motion += 1
            if int(num_people) > 0:
                true_pos += 1
        #Else no motion
        else:
            no_motion += 1
            if int(num_people) > 0:
                false_neg += 1
    
    print('True Positive of ',sensor,' :', true_pos / motion)
    print('False Negative of ',sensor,':', false_neg / no_motion)
    print('')

#Motion Censor
accuracy_motion('motion_sensor1','r1')
accuracy_motion('motion_sensor2','r2')
accuracy_motion('motion_sensor3','r3')
accuracy_motion('motion_sensor4','r4')
accuracy_motion('motion_sensor5','r5')
accuracy_motion('motion_sensor6','r6')
accuracy_motion('motion_sensor7','r7')
accuracy_motion('motion_sensor8','r8')
accuracy_motion('motion_sensor9','r9')
accuracy_motion('motion_sensor10','r10')

#Reliability of Camera Sensers

def accuracy_camera(sensor, room):
    #Count the number of times the camera actually spots someone
    correct = 0

    for i in range(len(sorted_data)):
        spotted = sorted_data[sensor].iloc[i]
        num_people = sorted_data[room].iloc[i]
        #If the camera spots someone and there exist someone
        if spotted > 0 and num_people > 0:
            correct += 1
        #If the camera spots no one and there is no one
        elif spotted == 0 and num_people == 0:
            correct += 1

    print('Accuracy for', sensor, ':', correct/(len(sorted_data)))
    print('\n')

#Camera
accuracy_camera('camera1','r1')
accuracy_camera('camera2','r4')
accuracy_camera('camera3','r8')
accuracy_camera('camera4','c3')

#Reliability of Door Sensers
def compute_door_sensor(sensor,room1,room2):
    #We need to know True positive and False Negative so we can appropriately change our data
    true_pos_r1 = 0
    false_neg_r1 = 0
    #We need to know True positive and False Negative so we can appropriately change our data
    true_pos_r2 = 0
    false_neg_r2 = 0

    #Total amount fo time the sensors is in motion
    motion = 0
    no_motion = 0

    for i in range(len(sorted_data)):
        #Number of people crossed
        observation = sorted_data[sensor].iloc[i]
        #Number of people in each room
        num_people_r1 = sorted_data[room1].iloc[i]
        num_people_r2 = sorted_data[room2].iloc[i]
        #If crossed
        if int(observation) > 0:
            motion += 1
            if int(num_people_r1) > 0:
                true_pos_r1 += 1
            if int(num_people_r2) > 0:
                true_pos_r2 += 1
        #Else no motion
        else:
            no_motion += 1
            if int(num_people_r1) > 0:
                false_neg_r1 += 1
            if int(num_people_r2) > 0:
                false_neg_r2 += 1
    
    print('True Positive of ', sensor, 'of ', room1, ': ', true_pos_r1 / motion)
    print('False Negative of ', sensor, 'of ', room1, ': ', false_neg_r1 / no_motion)
    print('True Positive of ', sensor, 'of ', room2, ': ', true_pos_r2 / motion)
    print('False Negative of ', sensor, 'of ', room2, ': ', false_neg_r2 / no_motion)
    print('')

#Door Sensor
compute_door_sensor('door_sensor1','r1','r2')
compute_door_sensor('door_sensor2','r3','c1')
compute_door_sensor('door_sensor3','r4','c1')
compute_door_sensor('door_sensor4','r4','r5')
compute_door_sensor('door_sensor5','r5','r6')
compute_door_sensor('door_sensor6','r9','c2')
compute_door_sensor('door_sensor7','r10','c2')
compute_door_sensor('door_sensor8','r7','r8')
compute_door_sensor('door_sensor9','r7','c3')
compute_door_sensor('door_sensor10','r2','c3')
compute_door_sensor('door_sensor11','r10','c3')