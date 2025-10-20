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
    
def stand_reward_smooth(state):
    standing_y = 500 - 93  # 407
    current_height = state['head']['position_b'].y
    
    perfect_height = standing_y + 20  # 427
    height_diff = abs(current_height - perfect_height)
    
    if height_diff <= 15:    # 412-442
        return 8.0 + 2.0 * (1 - height_diff / 15)
    elif height_diff <= 30:  # 397-457  
        return 4.0 + 4.0 * (1 - (height_diff - 15) / 15)
    elif height_diff <= 50:  # 377-477
        return 1.0 + 3.0 * (1 - (height_diff - 30) / 20)
    else:
        return 0.1
    
def stand_reward_strong(state):
    current_height = state['head']['position_b'].y
    ideal_height = 427
    
    height_diff = abs(current_height - ideal_height)
    
    if height_diff < 10:
        return 50.0
    elif height_diff < 30:
        return 20.0
    elif height_diff < 50:
        return 5.0
    else:
        return 0.1
    
def stand_reward_box2d(state, action):
    SCALE = 0.1
    
    shaping = (
        13 * (1 - state['torso']['pos'][1])
    )  # moving forward is a way to receive reward (normalized to get 300 on completion)
    shaping -= 0.5 * abs(
        state['torso']['angle']
    )  # keep head straight, other than that and falling, any behavior is unpunished

    return shaping

def action_penalty(state, action):
    import numpy as np
    reward = 0
    for a in action.values():
        reward -= 0.00035 * 80.0 * np.clip(np.abs(a), 0, 1)
    # normalized to about 50.0 using heuristic, more optimal agent should spend less

    return reward