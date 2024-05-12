import json
import random

from RUTAIGamesWebsocketHandler import RUTAIGamesWebsocketHandler


class ArkanoidWebSocketHandler(RUTAIGamesWebsocketHandler):
    def send_data(self, receivedData):
        move = random.choice([0, 1, -1])
        self.write_message(json.dumps({'aiMove': move}))
