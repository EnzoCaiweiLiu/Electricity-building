import pandas as pd

def compute_outcome_space(df):
    # Initialize outcomeSpace for each room
    outcome_space = {}
    for room in ['r' + str(i) for i in range(1, 11)] + ['c1', 'c2', 'c3', 'outside']:
        outcome_space[room] = []

    # Mapping door sensors to rooms
    door_sensor_map = {
        'door_sensor1': ('r1', 'r2'),
        'door_sensor2': ('r3', 'c1'),
        'door_sensor3': ('c1', 'r4'),
        'door_sensor4': ('r4', 'r5'),
        'door_sensor5': ('r5', 'r6'),
        'door_sensor6': ('c2', 'r9'),
        'door_sensor7': ('c2', 'r10'),
        'door_sensor8': ('r7', 'r8'),
        'door_sensor9': ('c3', 'r7'),
        'door_sensor10': ('r2', 'c3'),
        'door_sensor11': ('c3', 'r10'),
    }

    # Direct connections without sensors
    direct_connections = [
        ('r3', 'outside'),
        ('outside', 'r3')
    ]

    for connection in direct_connections:
        if connection[1] not in outcome_space[connection[0]]:
            outcome_space[connection[0]].append(connection[1])

    # Update outcomeSpace based on door sensor activations
    for _index, row in df.iterrows():
        for sensor, rooms in door_sensor_map.items():
            if row[sensor] > 0:
                if rooms[1] not in outcome_space[rooms[0]]:
                    outcome_space[rooms[0]].append(rooms[1])
                if rooms[0] not in outcome_space[rooms[1]]:
                    outcome_space[rooms[1]].append(rooms[0])

    # Connect c1 and c2 thoroughly due to boundary without sensors
    for room in list(outcome_space.keys()):
        if 'c1' in outcome_space[room]:
            outcome_space[room].append('c2')
        if 'c2' in outcome_space[room]:
            outcome_space[room].append('c1')

    # Remove duplicates
    for room, connections in outcome_space.items():
        outcome_space[room] = list(set(connections))

    return outcome_space

dataframe = pd.read_csv('data1.csv')
outcome_space_result = compute_outcome_space(dataframe)

print(outcome_space_result)
