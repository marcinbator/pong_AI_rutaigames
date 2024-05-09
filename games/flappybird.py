import json
import random

from RUTAIGamesWebsocketHandler import RUTAIGamesWebsocketHandler


class FlappyBirdWebSocketHandler(RUTAIGamesWebsocketHandler):
    def send_data(self, receivedData):
        move = random.choice([True, False])
        self.write_message(json.dumps({'aiUp': move}))
