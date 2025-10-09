import torch
import numpy as np

class RLAlgorithm:
    def __init__(self):
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state : np.array) -> np.array:
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def save(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError
    
    def update_check(self):
        return self.replay_buffer.full()
    
class ReplayBuffer():
    def __init__(self, max_size = 5000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def clear(self):
        self.storage = []

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def full(self):
        return len(self.storage) == self.max_size

    def sample(self, batch_size = -1):
        if batch_size == -1:
            ind = np.arange(0,len(self.storage))
        else:
            ind = np.random.randint(0,len(self.storage),size=batch_size).sort()
        x, y, u, r, d = [],[],[],[],[]

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(X)
            y.append(Y)
            u.append(U)
            r.append(R)
            d.append(D)

        return np.array(x), np.array(y), np.array(u), np.array(r), np.array(d)