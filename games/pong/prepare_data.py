import csv
import json

import numpy as np


def read_ball():
    balls_x = []
    balls_y = []
    balls_vel_x = []
    balls_vel_y = []
    with open("pong_pong.csv", 'r') as file:
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
                print("Nie udało się odczytać ostatniej kolumny jako JSON.")
    return balls_x, balls_y, balls_vel_x, balls_vel_y


data = np.loadtxt("pong_pong.csv", delimiter=',', dtype=str, usecols=(2, 12))

balls_x, balls_y, balls_vel_x, balls_vel_y = read_ball()
data = np.column_stack((balls_x, balls_y, balls_vel_x, balls_vel_y, data))

np.savetxt('prepared_' + 'pong_pong.csv', data, delimiter=',', fmt='%s')
