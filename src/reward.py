def stand_reward_zero(state):
    standing_y = 500 - 93
    tolerance = 20
    max_reward = 20
    return min(standing_y + tolerance - state['head']['position_b'].y, max_reward) * 0.001

def stand_reward(state):
    standing_y = 500 - 93
    tolerance = 20
    
    bias = state['head']['position_b'].y - (standing_y + tolerance)
    if bias < 0:
        return 1
    elif bias > 40:
        return 0
    else:
        return (50 - bias) * 0.001