from app.core import Child, MAX_RETRIES
from app import util


def start():
    c = Child()
    return c.run(), util.whisper("hi"), MAX_RETRIES