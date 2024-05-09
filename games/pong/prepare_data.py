import csv
import json

import numpy as np

import csv
import json
import numpy as np
import pandas as pd

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

# Zamiana typów danych na numeryczne dla wszystkich kolumn
df = df.apply(pd.to_numeric, errors='ignore')

# Definiowanie funkcji do grupowania
def custom_round(x):
    return round(x / 3) * 3

# Zaokrąglanie każdej z pierwszych pięciu kolumn
for i in range(5):
    df[i] = df[i].apply(custom_round)

# Grupowanie i obliczanie średniej dla pierwszych pięciu kolumn oraz modę dla szóstej kolumny
df_grouped = df.groupby(list(df.columns[:5])).agg({0: 'mean', 1: 'mean', 2: 'mean', 3: 'mean', 4: 'mean', 5: lambda x: pd.Series.mode(x)[0]})

df_grouped.to_csv('prepared_pong_pong.csv', index=False, header=False)