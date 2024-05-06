import csv
import json

import numpy as np

import numpy as np


def generate_board(snake_data, apple_data_y, apple_data_x, head_y, head_x):
    board = [0] * 20 * 20

    for segment in snake_data:
        x, y = segment['positionX'], segment['positionY']
        board[y * 20 + x] = -1

    board[apple_data_y * 20 + apple_data_x] = 2
    board[head_y * 20 + head_x] = 1

    return board


def read_tail(filename):
    tails = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            json_str = row[-1].strip('"')
            try:
                json_list = json.loads(json_str)
                tails.append(json_list)
            except json.JSONDecodeError:
                print("Nie udało się odczytać ostatniej kolumny jako JSON.")
    return tails


def map_labels_to_numbers(data):
    labels_column_6 = data[:, 6]
    label_map = {'up': 1, 'down': 2, 'left': 3, 'right': 4}

    mapped_labels_6 = np.array([label_map[label] if label in label_map else label for label in labels_column_6])

    data[:, 6] = mapped_labels_6.astype(str)

    return data, label_map


def prepare_data(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=str, usecols=range(7))
    tails = read_tail(filename)
    boards = []
    for i in range(len(tails)):
        boards.append(generate_board(tails[i], int(data[i][2]), int(data[i][1]), int(data[i][4]), int(data[i][3])))
    boards_array = np.array(boards)
    data = np.column_stack((data, boards_array))
    data, label_map = map_labels_to_numbers(data)
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 0, axis=1)
    data = np.delete(data, 0, axis=1)
    data = np.unique(data, axis=0)
    np.savetxt('prepared_' + filename, data, delimiter=',', fmt='%s')


prepare_data("snake_snake.csv")
prepare_data("snake_snake_test.csv")
