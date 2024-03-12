# pylint: skip-file

#Import necessary file
import numpy as np
import pandas as pd

#Get data
data_1 = pd.read_csv('data1.csv')
data_2 = pd.read_csv('data2.csv')

#Combine data for easy iteration
extended_data = pd.concat([data_1, data_2], ignore_index=True)
data = extended_data.sort_values(by='time')

#Dense network model
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

def count_tran(data, key):
    # Convert the 'key' data from 'data' into a list
    list1 = list(data[key])

    # Create a new list 'd2' that contains all elements from 'd1' except the first one
    list2 = list1[1:]

    # Add a '0' at the end of 'd2' to make it the same length as 'd1'
    list2.append(0)

    # Convert 'd1' and 'd2' to NumPy arrays
    data1 = np.array(list1)
    data2 = np.array(list2)

    # Check if the sum of 'data1' is equal to 0
    if data1.sum() == 0:
        return 0

    # Calculate the difference between 'data1' and 'data2'
    diff = data1 - data2

    # Initialize a variable 'sum' to 0, which will be used to accumulate the positive differences
    sum = 0

    # Iterate through each element 'once' in the 'diff' array
    for once in diff:
        if once > 0:
            # If the element is positive, add it to 'sum'
            sum += once

    # Return the ratio of the accumulated positive differences to the sum of 'd1'
    return sum / data1.sum()

pos = [0,38,4558,len(data)]
index = {}
start = 0
for key in outcomeSpace.keys():
    index[key] = start
    start += 1

#3 time period transaction matrix, will be stored in tran_matrix_probability.csv file.
output = {}

#We figured out that there was drastic different in the movement of people at the morning, normal working hours, leaving
for loc in range(len(pos) - 1):
    #(8:00 - 8:05) / (8:05 - 17:30) / (17:30 - 18:00)
    for key in outcomeSpace.keys():
        #The transition probability of the location
        trans = [0]*14

        #Get the possiblity of moving to the other rooms (specific)
        for i in range(len(data[pos[loc] : pos[loc + 1]]) - 1):
            current_data = data.iloc[pos[loc] + i]
            future_data = data.iloc[pos[loc] + i + 1]

            #If someone move out of the room
            if int(future_data[key]) < int(current_data[key]):

                #Get the difference
                diff = int(current_data[key]) - int(future_data[key])

                #Go through all room to find its possible location
                for possible_mov in outcomeSpace[key]:
                    #Num of people in the future and now of that particular room
                    num_current = current_data[possible_mov]
                    num_future = future_data[possible_mov]

                    #Get the difference of that room
                    diff_other = int(num_future) - int(num_current)

                    #If its greater than the difference of the original room we assume all of it goes here
                    if diff_other >= diff:
                        #Add it to that location
                        trans[index[possible_mov]] += diff
                        break
                    #Otherwise
                    else:
                        #If there is a positive difference
                        if diff_other > 0:
                            #Assume that a percantage of people goes here
                            trans[index[possible_mov]] += diff_other
                            diff -= diff_other
        
        #Turn that data into a list
        list1 = list(data[pos[loc] : pos[loc + 1]][key])
        data1 = np.array(list1)

        #Turn it into probability instead of occurance
        if data1.sum() != 0:
            trans[:] = [x/data1.sum() for x in trans]
        
        #Get the negative for the opposite
        trans[index[key]] = 1 - count_tran(data[pos[loc] : pos[loc + 1]], key)
        #Put it into output
        if loc == 0 :
            output[key+'_time1'] = trans
        elif loc == 1 :
            output[key+'_time2'] = trans
        elif loc == 2 :
            output[key+'_time3'] = trans

#output to csv file.
state_csv = pd.DataFrame(output)
state_csv.to_csv('tran_matrix_probability.csv',index=False)