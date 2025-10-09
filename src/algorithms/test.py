from PPO import RolloutBuffer
import numpy as np

if __name__ == '__main__':
    buf = RolloutBuffer(5)

    for i in range(6):
        a = np.ones(24) * i,
        b = np.ones(4) * i
        c = np.ones(1) * i
        buf.push(a, b, c, c, c, c)

    for batch in buf.get(2):
        print(batch)