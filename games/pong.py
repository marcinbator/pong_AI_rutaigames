import json

from RUTAIGamesWebsocketHandler import RUTAIGamesWebsocketHandler


class PongWebSocketHandler(RUTAIGamesWebsocketHandler):
    def send_data(self, receivedData):
        move = 0
        if receivedData[1][1]['value'] > receivedData[1][3]['value']:
            move = 1
        elif receivedData[1][1]['value'] < receivedData[1][3]['value']:
            move = -1
        self.write_message(json.dumps({'up': move}))
