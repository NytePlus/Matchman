import os
import sys

from src.deploy.backend import backend

app = backend.app
socketio = backend.socketio