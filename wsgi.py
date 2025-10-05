from threading import Thread
from src.deploy.backend import backend, trainer, static

Thread(target=trainer.train).start()
static.start()

app = backend.app
socketio = backend.socketio
