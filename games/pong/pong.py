import json

from RUTAIGamesWebsocketHandler import RUTAIGamesWebsocketHandler
from games.pong.model import predict_result


class PongWebSocketHandler(RUTAIGamesWebsocketHandler):
    def send_data(self, receivedData):
        print(receivedData)
        data = [
            receivedData[1][0]['value']['x'],
            receivedData[1][0]['value']['y'],
            receivedData[1][0]['value']['velocityX'],
            receivedData[1][0]['value']['velocityY'],
            receivedData[1][2]['value'],
        ]
        print("data", data)
        move = predict_result(data)
        print("move",move)
        self.write_message(json.dumps({'up': int(move)}))
