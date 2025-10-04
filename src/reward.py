def stand_reward(state):
    standing_y = 500 - 93
    tolerance = 20
    max_reward = 20
    return min(standing_y + tolerance - state['head']['position_a'].y, max_reward)