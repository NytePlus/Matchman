import pymunk
import pygame
import pymunk.pygame_util
import numpy as np

MATCHMAN_CATEGORY = 0b0001
GROUND_CATEGORY = 0b0010

# --- env-network protocol ---
def pack_state(state_dict : dict):
    state = []
    for key, value in state_dict.items():
        for _, v in value.items():
            if isinstance(v, pymunk.Vec2d):
                state.extend([v.x, v.y])
            else:
                state.append(v)
    state = np.array(state)
    return state

def unpack_action(action : np.array):
    action_dict = {
        "head": action[0],
        "left_arm": action[1],
        "left_forearm": action[2],
        "right_arm": action[3],
        "right_forearm": action[4],
        "left_leg": action[5],
        "left_foreleg": action[6],
        "right_leg": action[7],
        "right_foreleg": action[8],
    }
    return action_dict

# --- env initialization ---
def create_ground(space):
    ground = pymunk.Segment(space.static_body, (-100, 500), (1000, 500), 5)
    ground.friction = 1.0
    ground.filter = pymunk.ShapeFilter(categories=GROUND_CATEGORY, mask=MATCHMAN_CATEGORY)
    space.add(ground)

    left_wall = pymunk.Segment(space.static_body, (0, 500), (0, -500), 5)
    left_wall.friction = 1.0
    left_wall.filter = pymunk.ShapeFilter(categories=GROUND_CATEGORY, mask=MATCHMAN_CATEGORY)
    space.add(left_wall)

    right_wall = pymunk.Segment(space.static_body, (800, 500), (800, -500), 5)
    right_wall.friction = 1.0
    right_wall.filter = pymunk.ShapeFilter(categories=GROUND_CATEGORY, mask=MATCHMAN_CATEGORY)
    space.add(right_wall)

def create_segment(space, pos, mass, a, b, fric=0.7, radius=3):
    moment = pymunk.moment_for_segment(mass, a, b, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Segment(body, a, b, radius)
    shape.friction = fric
    shape.filter = pymunk.ShapeFilter(categories=MATCHMAN_CATEGORY, mask=GROUND_CATEGORY)
    space.add(body, shape)

    return body

def create_matchman(space, pos):
    # 躯干
    torso_size = 50
    torso_body = create_segment(space, pos, 1, (0, torso_size / 2), (0, -torso_size / 2))

    # 头部
    head_mass = 0.5
    head_radius = 15
    head_moment = pymunk.moment_for_circle(head_mass, 0, head_radius)
    head_body = pymunk.Body(head_mass, head_moment)
    head_body.position = (pos[0], pos[1] - torso_size / 2 - head_radius)
    head_shape = pymunk.Circle(head_body, head_radius)
    head_shape.friction = 0.7
    head_shape.filter = pymunk.ShapeFilter(categories=MATCHMAN_CATEGORY, mask=GROUND_CATEGORY)
    space.add(head_body, head_shape)

    # 连接头部和躯干
    head_joint = pymunk.PinJoint(torso_body, head_body, (0, -torso_size / 2), (0, 15))
    space.add(head_joint)

    # 控制头部旋转
    head_motor = pymunk.SimpleMotor(torso_body, head_body, 0)  # 初始角速度为 0
    space.add(head_motor)

    # 左大臂
    left_arm_body = create_segment(space, (pos[0] - 7, pos[1] - torso_size / 2 - 7), 0.3, (7, 7), (-7, -7))

    # 连接左大臂和躯干
    left_arm_joint = pymunk.PinJoint(torso_body, left_arm_body, (0, -torso_size / 2), (7, 7))
    space.add(left_arm_joint)

    # 控制左大臂旋转
    left_arm_motor = pymunk.SimpleMotor(torso_body, left_arm_body, 0)
    space.add(left_arm_motor)

    # 左小臂
    left_forearm_body = create_segment(space, (pos[0] - 21, pos[1] - torso_size / 2 - 21), 0.3, (7, 7), (-7, -7))

    # 连接左小臂和左大臂
    left_forearm_joint = pymunk.PinJoint(left_arm_body, left_forearm_body, (- 7, - 7), (7, 7))
    space.add(left_forearm_joint)

    # 控制左小臂旋转
    left_forearm_motor = pymunk.SimpleMotor(left_arm_body, left_forearm_body, 0)
    space.add(left_forearm_motor)

    # 右大臂
    right_arm_body = create_segment(space, (pos[0] + 7, pos[1] - torso_size / 2 - 7), 0.3, (7, -7), (-7, 7))

    # 连接右大臂和躯干
    right_arm_joint = pymunk.PinJoint(torso_body, right_arm_body, (0, -torso_size / 2), (-7, 7))
    space.add(right_arm_joint)

    # 控制右大臂旋转
    right_arm_motor = pymunk.SimpleMotor(torso_body, right_arm_body, 0)
    space.add(right_arm_motor)

    # 右小臂
    right_forearm_body = create_segment(space, (pos[0] + 21, pos[1] - torso_size / 2 - 21), 0.3, (7, -7), (-7, 7))

    # 连接右小臂和右大臂
    right_forearm_joint = pymunk.PinJoint(right_arm_body, right_forearm_body, (7, - 7), (-7, 7))
    space.add(right_forearm_joint)

    # 控制右小臂旋转
    right_forearm_motor = pymunk.SimpleMotor(right_arm_body, right_forearm_body, 0)
    space.add(right_forearm_motor)

    # 左大腿
    left_leg_body = create_segment(space, (pos[0] - 7, pos[1] + torso_size / 2 + 14), 0.5, (7, -14), (-7, 14))

    # 连接左大腿和躯干
    left_leg_joint = pymunk.PinJoint(torso_body, left_leg_body, (0, torso_size / 2), (7, -14))
    space.add(left_leg_joint)

    # 控制左大腿旋转
    left_leg_motor = pymunk.SimpleMotor(torso_body, left_leg_body, 0)
    space.add(left_leg_motor)

    # 左小腿
    left_foreleg_body = create_segment(space, (pos[0] - 21, pos[1] + torso_size / 2 + 42), 0.5, (7, -14), (-7, 14))

    # 连接左小腿和左大腿
    left_foreleg_joint = pymunk.PinJoint(left_leg_body, left_foreleg_body, (-7, 14), (7, -14))
    space.add(left_foreleg_joint)

    # 控制左小腿旋转
    left_foreleg_motor = pymunk.SimpleMotor(left_leg_body, left_foreleg_body, 0)
    space.add(left_foreleg_motor)

    # 右大腿
    right_leg_body = create_segment(space, (pos[0] + 7, pos[1] + torso_size / 2 + 14), 0.5, (7, 14), (-7, -14))

    # 连接右大腿和躯干
    right_leg_joint = pymunk.PinJoint(torso_body, right_leg_body, (0, torso_size / 2), (-7, -14))
    space.add(right_leg_joint)

    # 控制右大腿旋转
    right_leg_motor = pymunk.SimpleMotor(torso_body, right_leg_body, 0)
    space.add(right_leg_motor)

    # 右小腿
    right_foreleg_body = create_segment(space, (pos[0] + 21, pos[1] + torso_size / 2 + 42), 0.5, (7, 14), (-7, -14))

    # 连接右小腿和右大腿
    right_foreleg_joint = pymunk.PinJoint(right_leg_body, right_foreleg_body, (7, 14), (-7, -14))
    space.add(right_foreleg_joint)

    # 控制右小腿旋转
    right_foreleg_motor = pymunk.SimpleMotor(right_leg_body, right_foreleg_body, 0)
    space.add(right_foreleg_motor)

    bodys = {
        "torso": torso_body,
        "head": head_body,
        "left_arm": left_arm_body,
        "left_forearm": left_forearm_body,
        "right_arm": right_arm_body,
        "right_forearm": right_forearm_body,
        "left_leg": left_leg_body,
        "left_foreleg": left_foreleg_body,
        "right_leg": right_leg_body,
        "right_foreleg": right_foreleg_body,
    }

    motors = {
        "head": head_motor,
        "left_arm": left_arm_motor,
        "left_forearm": left_forearm_motor,
        "right_arm": right_arm_motor,
        "right_forearm": right_forearm_motor,
        "left_leg": left_leg_motor,
        "left_foreleg": left_foreleg_motor,
        "right_leg": right_leg_motor,
        "right_foreleg": right_foreleg_motor,
    }
    return motors, bodys

# -- env runtime ---
def get_body_endpoints(body):
    def get_segment_endpoints(body, segment_shape):
            world_a = body.local_to_world(segment_shape.a)
            world_b = body.local_to_world(segment_shape.b)
            return world_a, world_b

    def get_circle_endpoints(body, circle_shape):
        radius = circle_shape.radius
        world_a = body.local_to_world(pymunk.Vec2d(-radius, 0) + circle_shape.offset)
        world_b = body.local_to_world(pymunk.Vec2d(radius, 0) + circle_shape.offset)
        return world_a, world_b
    
    shapes = [s for s in body.shapes]
    shape = shapes[0]
    if isinstance(shape, pymunk.Segment):
        return get_segment_endpoints(body, shape)
    elif isinstance(shape, pymunk.Circle):
        return get_circle_endpoints(body, shape)
    else:
        raise ValueError("Unsupported shape type for body endpoints")

def get_body_states(bodys):
    states = {}
    for name, body in bodys.items():
        a, b = get_body_endpoints(body)

        states[name] = {
            "position_a": a,
            "position_b": b,
        }
    return states

def get_joint_states(motors):
    states = {}
    for name, motor in motors.items():
        # 获取的是下游的躯干，因为总是“身体-大臂”，“大臂-小臂”
        body = motor.b

        a, b = get_body_endpoints(body) # 获取关节两个端点
        vel = body.velocity # 获取关节的速度
        angle = body.angle 
        angular_vel = body.angular_velocity # 获取关节的角度和角速度        
        motor_rate = motor.rate # 获取角动量

        states[name] = {
            "position_a": a,
            "position_b": b,
            "velocity_b": vel,
            "angle_b": angle,
            "angular_velocity_b": angular_vel,
            "motor_rate": motor_rate,
        }

    return states

def get_states(motors, bodys):
    return {
        'motors': get_joint_states(motors),
        'bodys': get_body_states(bodys),
    }

def set_motor_rates(motors, rates):
    for name, rate in rates.items():
        motors[name].rate = rate

# --- env operation ---
class MatchmanEnv():
    def __init__(self, rewards, draw = False):
        self.rewards = rewards
        self._running = False
        self.space = None
        self.draw = draw

        if draw:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def reset(self):
        self._running = True
        self.space = pymunk.Space()
        self.space.gravity = (0, 200)

        create_ground(self.space)
        self.motors, self.bodys = create_matchman(self.space, (400, 300))

        return get_joint_states(self.motors)

    def running(self):
        return self._running

    def step(self, action, test = False):
        if self.draw and test:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    pygame.quit()
                    return None, None, None
            self.screen.fill((255, 255, 255))
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            self.clock.tick(6000)
        self.space.step(1/30)

        set_motor_rates(self.motors, action)

        next_state = get_joint_states(self.motors)
        reward = sum([r(next_state) for r in self.rewards])
        done = False
        return next_state, reward, done
    
    def get_state(self):
        motor_state = get_joint_states(self.motors)
        body_state = get_body_states(self.bodys)

        return motor_state, body_state