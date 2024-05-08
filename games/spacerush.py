import json
import random

from RUTAIGamesWebsocketHandler import RUTAIGamesWebsocketHandler


class SpaceRushWebSocketHandler(RUTAIGamesWebsocketHandler):
    def send_data(self, receivedData):
        keys = ['up', 'down', 'left', 'right', 'none']
        shot = [True, False]
        keyIndex = random.randint(0, len(keys) - 1)
        shotIndex = random.randint(0, len(shot) - 1)
        self.write_message(json.dumps({'aiMoves': [keys[keyIndex]], 'aiShot': shot[shotIndex]}))
