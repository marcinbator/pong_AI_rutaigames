import csv
import json
import numpy as np

name = "pong_pong.csv"

def read_ball():
    balls_x = []
    balls_y = []
    balls_vel_x = []
    balls_vel_y = []
    with open(name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            json_str = row[-3].strip('"')
            try:
                json_list = json.loads(json_str)
                balls_x.append(round(json_list['x'], 1))
                balls_y.append(round(json_list['y'], 1))
                balls_vel_x.append(round(json_list['velocityX'], 1))
                balls_vel_y.append(round(json_list['velocityY'], 1))
            except json.JSONDecodeError:
                print("Failed to read the last column as JSON.")
    return balls_x, balls_y, balls_vel_x, balls_vel_y

def convert_label(value):
    if value == '-1':
        return 'down'
    elif value == '0':
        return 'stay'
    elif value == '1':
        return 'up'
    else:
        return 'unknown'

data = np.loadtxt(name, delimiter=',', dtype=str, usecols=(2, 12))

balls_x, balls_y, balls_vel_x, balls_vel_y = read_ball()
data = np.column_stack((balls_x, balls_y, balls_vel_x, balls_vel_y, data))

# Convert the last column to labels
data[:, -1] = np.vectorize(convert_label)(data[:, -1])

np.savetxt('etiquettes_prepared_pong_pong.csv', data, delimiter=',', fmt='%s')
