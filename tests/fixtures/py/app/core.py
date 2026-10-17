"""Core helpers."""
from .util import shout

MAX_RETRIES = 3


class Base:
    def run(self):
        return shout("x")


class Child(Base):
    def run(self):
        return super().run() + 1

    def other(self):
        return self.run()