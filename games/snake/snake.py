import json

from RUTAIGamesWebsocketHandler import RUTAIGamesWebsocketHandler
from games.snake.model import predict_result


def generate_board(snake_data, apple_data_y, apple_data_x, head_data_y, head_data_x):
    board = [0] * 20 * 20

    for segment in snake_data['value']:
        x, y = segment['positionX'], segment['positionY']
        board[y * 20 + x] = -1

    apple_x, apple_y = apple_data_x['value'], apple_data_y['value']
    head_x, head_y = head_data_x['value'], head_data_y['value']
    board[apple_y * 20 + apple_x] = 2
    board[head_y * 20 + head_x] = 1

    # Wypisywanie planszy w 10 rzędach po 10
    for i in range(20):
        for j in range(20):
            print(board[i * 20 + j], end=" ")
        print()
    return board


class SnakeWebSocketHandler(RUTAIGamesWebsocketHandler):
    def send_data(self, receivedData):
        data = []
        # for i in range(4):
        #     data.append(receivedData[1][i]['value'])

        for e in generate_board(receivedData[1][4], receivedData[1][1], receivedData[1][0], receivedData[1][3],
                                receivedData[1][2]):
            data.append(e)

        print(data)
        keys = ['up', 'down', 'left', 'right']
        keyIndex = predict_result(data) - 1
        print("selected: ", keys[keyIndex])
        self.write_message(json.dumps({'key': keys[keyIndex]}))
