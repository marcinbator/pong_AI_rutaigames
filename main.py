import tornado.ioloop
import tornado.web
import tornado.websocket

from games.pong.pong import PongWebSocketHandler
from games.snake.snake import SnakeWebSocketHandler
from games.spacerush import SpaceRushWebSocketHandler
from games.tictactoe import TicTacToeWebSocketHandler
from games.flappybird import FlappyBirdWebSocketHandler
from games.arkanoid import ArkanoidWebSocketHandler


def make_app():
    return tornado.web.Application([
        (r"/ws/snake/", SnakeWebSocketHandler),
        (r"/ws/pong/", PongWebSocketHandler),
        (r"/ws/tictactoe/", TicTacToeWebSocketHandler),
        (r"/ws/spacerush/", SpaceRushWebSocketHandler),
        (r"/ws/flappybird/", FlappyBirdWebSocketHandler),
        (r"/ws/arkanoid/", ArkanoidWebSocketHandler),
    ])


def run():
    app = make_app()
    app.listen(8001)
    print("Started")
    tornado.ioloop.IOLoop.current().start()


run()
