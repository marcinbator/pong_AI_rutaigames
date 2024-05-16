import csv
import json

import numpy as np
import pandas as pd

name = "pong_pong_normal.csv"


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
                balls_x.append(round(json_list['x'], 0))
                balls_y.append(round(json_list['y'], 0))
                balls_vel_x.append(round(json_list['velocityX'], 0))
                balls_vel_y.append(round(json_list['velocityY'], 0))
            except json.JSONDecodeError:
                print("Nie udało się odczytać ostatniej kolumny jako JSON.")
    return balls_x, balls_y, balls_vel_x, balls_vel_y


data = np.loadtxt(name, delimiter=',', dtype=str, usecols=(2, 12))

balls_x, balls_y, balls_vel_x, balls_vel_y = read_ball()
data = np.column_stack((balls_x, balls_y, balls_vel_x, balls_vel_y, data))

df = pd.DataFrame(data)

df.to_csv('output/prepared_pong_pong_normal.csv', index=False, header=False)
