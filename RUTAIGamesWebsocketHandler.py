import json

import tornado.websocket


class RUTAIGamesWebsocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("WebSocket connection opened")

    def on_message(self, message):
        print(message)
        receivedData = json.loads(message)
        self.send_data(receivedData)

    def on_close(self):
        print("WebSocket connection closed")

    def send_data(self, receivedData):
        raise NotImplementedError("send_data method must be implemented in subclass")
