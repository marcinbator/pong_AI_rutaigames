import json
import random

from RUTAIGamesWebsocketHandler import RUTAIGamesWebsocketHandler


class TicTacToeWebSocketHandler(RUTAIGamesWebsocketHandler):
    def send_data(self, receivedData):
        available = receivedData[1][0]['value']
        if len(available) > 0:
            keyIndex = random.randint(0, len(available) - 1)
            self.write_message(json.dumps({'aiMoveIndex': keyIndex}))
