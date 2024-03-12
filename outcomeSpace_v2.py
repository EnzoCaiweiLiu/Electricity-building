import pandas as pd

# 读取CSV文件到DataFrame
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')

# 楼层图数据结构
graph_structure = {
    'r1': ['r2'], 
    'r2': ['r1', 'c3'], 
    'r3': ['outside', 'c1', 'c2'], 
    'r4': ['r5', 'c1', 'c2'], 
    'r5': ['r6', 'r4'], 
    'r6': ['r5'], 
    'r7': ['c3', 'r8'], 
    'r8': ['r7'], 
    'r9': ['c1', 'c2'], 
    'r10': ['c3', 'c1', 'c2'], 
    'c1': ['r3', 'r4', 'r9'], 
    'c2': ['r3', 'r4', 'r9', 'r10'], 
    'c3': ['r2', 'r7', 'r10'], 
    'outside': ['r3']
}

# 这里的sensor_data是一个字典，包含了当前时间点的所有传感器读数
def get_potential_destinations(sensor_data, graph):
    potential_destinations = {room: set() for room in graph}
    for room, connected_rooms in graph.items():
        motion_sensor = sensor_data.get(f'motion_sensor{room[1:]}')
        door_sensor = sensor_data.get(f'door_sensor{room[1:]}', 0)
        if motion_sensor == 'motion' or door_sensor > 0:
            potential_destinations[room].update(connected_rooms)
        # 检查机器人数据
        for robot_id in ['robot1', 'robot2']:
            robot_data = sensor_data.get(robot_id)
            if robot_data:
                robot_room, count = eval(robot_data)  # 这里使用eval函数将字符串转换为元组
                if robot_room == room and count > 0:
                    potential_destinations[room].update(connected_rooms)
    return potential_destinations

def calculate_potential_destinations(df, graph):
    potential_destinations_time = {}
    for index, row in df.iterrows():
        sensor_data = row.to_dict()
        potential_destinations = get_potential_destinations(sensor_data, graph)
        potential_destinations_time[sensor_data['time']] = potential_destinations
    return potential_destinations_time

# 计算潜在目的地
potential_destinations_data1 = calculate_potential_destinations(data1, graph_structure)
potential_destinations_data2 = calculate_potential_destinations(data2, graph_structure)

# 打印结果的第一个条目
print('Data1 First Entry Potential Destinations:')
print(potential_destinations_data1[next(iter(potential_destinations_data1))])

print('Data2 First Entry Potential Destinations:')
print(potential_destinations_data2[next(iter(potential_destinations_data2))])