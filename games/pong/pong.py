import json

from RUTAIGamesWebsocketHandler import RUTAIGamesWebsocketHandler
# from games.pong.algorithm.model import predict_result as algorithm
# from games.pong.normal.model import predict_result as normal
from games.pong.normal_cnn.model import predict_result as normal

class PongWebSocketHandler(RUTAIGamesWebsocketHandler):
    def send_data(self, receivedData):
        data = [
            receivedData[1][0]['value']['x'],
            receivedData[1][0]['value']['y'],
            receivedData[1][0]['value']['velocityX'],
            receivedData[1][0]['value']['velocityY'],
            receivedData[1][2]['value'],
        ]
        print(data)
        move = normal(data)
        print("move: ", move)
        self.write_message(json.dumps({'up': int(move)}))
