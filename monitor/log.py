import logging
from .singleton import Singleton


class Logger(Singleton):

    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
