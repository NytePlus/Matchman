import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Box2D import *

import pygame 

# --- env constant --- 
# Box2D使用MKS单位制 (米, 千克, 秒)。我们将尺寸从原有的PyMunk抽象单位调整为MKS
SCALE = 0.1  # 10 original units = 1 meter
TORSO_SIZE = 50 * SCALE  # 5.0 meters
HEAD_RADIUS = 15 * SCALE
GROUND_Y = 500 * SCALE
LEG_HALF_SIZE = 14 * SCALE
ARM_HALF_SIZE = 10 * SCALE
WIDTH = 3 * SCALE

# 扭矩控制参数
MAX_TORQUE = 800.0
TARGET_SPEED = 10.0

# 关节角度限制 (弧度，与PyMunk保持一致)
HIP_MIN = np.deg2rad(-100)
HIP_MAX = np.deg2rad(70)
KNEE_MIN = np.deg2rad(-170)
KNEE_MAX = np.deg2rad(170)
HEAD_MIN = np.deg2rad(-50)
HEAD_MAX = np.deg2rad(50)

limb_half_size = {
    "torso": TORSO_SIZE / 2,
    "head": HEAD_RADIUS,
    "left_arm": ARM_HALF_SIZE,
    "left_forearm": ARM_HALF_SIZE,
    "right_arm": ARM_HALF_SIZE,
    "right_forearm": ARM_HALF_SIZE,
    "left_leg": LEG_HALF_SIZE,
    "right_leg": LEG_HALF_SIZE,
    "left_foreleg": LEG_HALF_SIZE,
    "right_foreleg": LEG_HALF_SIZE,
}

NO_COLISION_GROUP = -1
GROUND_CATEGORY = 0x0001
BODY_CATEGORY = 0x0002

# --- env-network protocol ---
def pack_state(state_dict : dict):
    state = []

    for key, value in state_dict.items():
        for _, v in value.items():
            if isinstance(v, (list, tuple)):
                state.extend(v)
            else:
                state.append(v)
    state = np.array(state)
    return state

def unpack_action(action : np.array):
    # 动作顺序不变 (9维)
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

def create_segment_box2d(world, pos, mass, half_height, angle=0):
    body_def = b2BodyDef(
        type=b2_dynamicBody,
        position=b2Vec2(pos[0], pos[1]),
        angle=angle
    )
    body = world.CreateBody(body_def)
    
    shape = b2PolygonShape(box=(WIDTH, half_height))
    fixture = body.CreateFixture(
        shape=shape, 
        density=mass / (WIDTH * 2 * half_height), 
        friction=0.7, 
        filter=b2Filter(
            groupIndex=NO_COLISION_GROUP, 
            categoryBits=BODY_CATEGORY,
            maskBits=GROUND_CATEGORY)
        )
    
    return body

def create_boundaries(world, width, height):
    def compute_fixture(body, shape):
        fixture_def = b2FixtureDef(
            shape=shape,
            density=0.0,
            friction=0.7,
            filter=b2Filter(categoryBits=GROUND_CATEGORY)
        )
        body.CreateFixture(fixture_def)

    ground_body = world.CreateBody(
        position=(0, GROUND_Y),
        type=b2_staticBody
    )
    ground_shape = b2PolygonShape(box=(1000 * SCALE, WIDTH))
    compute_fixture(ground_body, ground_shape)
    
    
    left_wall_pos_x = WIDTH / 2 
    left_wall_pos_y = GROUND_Y + height / 2
    
    left_wall_body = world.CreateBody(
        position=(left_wall_pos_x, left_wall_pos_y),
        type=b2_staticBody
    )
    left_wall_shape = b2PolygonShape(box=(WIDTH / 2, height / 2 + GROUND_Y)) 
    compute_fixture(left_wall_body, left_wall_shape)

    right_wall_pos_x = width - WIDTH / 2
    right_wall_pos_y = GROUND_Y + height / 2
    
    right_wall_body = world.CreateBody(
        position=(right_wall_pos_x, right_wall_pos_y),
        type=b2_staticBody
    )
    right_wall_shape = b2PolygonShape(box=(WIDTH / 2, height / 2 + GROUND_Y))
    compute_fixture(right_wall_body, right_wall_shape)
    
    return [ground_body, left_wall_body, right_wall_body]

def create_matchman(world, pos):
    motors, bodys = dict(), dict()
    
    # 躯干
    torso_half_size = TORSO_SIZE / 2
    torso_body = create_segment_box2d(world, pos, 0.5 * SCALE, torso_half_size) # 宽度 0.1m, 高度 5m, 质量 10kg
    bodys["torso"] = torso_body

    # 头部 (圆形)
    head_mass = 1
    head_body_def = b2BodyDef(type=b2_dynamicBody, position=b2Vec2(pos[0], pos[1] - torso_half_size - HEAD_RADIUS))
    head_body = world.CreateBody(head_body_def)
    head_shape = b2CircleShape(radius=HEAD_RADIUS)
    head_body.CreateFixture(shape=head_shape, density=head_mass / (np.pi * HEAD_RADIUS**2), friction=0.7)
    bodys["head"] = head_body
    
    # --- 关节辅助函数 ---
    def create_revolute_joint(body_a, body_b, anchor_a, anchor_b, lower_limit=None, upper_limit=None, name=""):
        jd = b2RevoluteJointDef()
        
        # anchor_a, anchor_b 是局部坐标系中的锚点
        # RevoluteJointDef.Initialize 期望世界坐标系的连接点
        jd.Initialize(body_a, body_b, body_a.GetWorldPoint(anchor_a))
        
        # 启用电机和限制
        jd.enableMotor = True
        jd.maxMotorTorque = MAX_TORQUE # 初始值，会被 step 函数覆盖
        
        if lower_limit is not None and upper_limit is not None:
            jd.enableLimit = True
            jd.lowerAngle = lower_limit
            jd.upperAngle = upper_limit
        
        # 创建并返回关节
        joint = world.CreateJoint(jd)
        motors[name] = joint
        return joint

    # --- 头部 (躯干连接) --- 
    create_revolute_joint(
        torso_body, head_body, b2Vec2(0, -torso_half_size), b2Vec2(0, HEAD_RADIUS), 
        HEAD_MIN, HEAD_MAX, "head"
    )

    # --- 腿部 (简化为矩形) ---
    HIP_ANCHOR = b2Vec2(0, TORSO_SIZE / 2) # 躯干底部中心

    # 左大腿
    left_leg_pos = b2Vec2(pos[0], pos[1] + torso_half_size + LEG_HALF_SIZE)
    left_leg_body = create_segment_box2d(world, left_leg_pos, 5, LEG_HALF_SIZE)
    bodys["left_leg"] = left_leg_body
    # 左髋关节
    create_revolute_joint(
        torso_body, left_leg_body, HIP_ANCHOR, b2Vec2(0, -LEG_HALF_SIZE), 
        HIP_MIN, HIP_MAX, "left_leg"
    )
    
    # 左小腿
    left_foreleg_pos = b2Vec2(pos[0], left_leg_pos.y + LEG_HALF_SIZE + LEG_HALF_SIZE)
    left_foreleg_body = create_segment_box2d(world, left_foreleg_pos, 4, LEG_HALF_SIZE)
    bodys["left_foreleg"] = left_foreleg_body
    # 左膝关节 (连接点: 大腿底部中心，小腿顶部中心)
    KNEE_ANCHOR_LEG = b2Vec2(0, LEG_HALF_SIZE) # 大腿底部
    KNEE_ANCHOR_FORELEG = b2Vec2(0, -LEG_HALF_SIZE) # 小腿顶部
    create_revolute_joint(
        left_leg_body, left_foreleg_body, KNEE_ANCHOR_LEG, KNEE_ANCHOR_FORELEG, 
        KNEE_MIN, KNEE_MAX, "left_foreleg"
    )

    # 右大腿 (使用负的关节限制，以确保对称性)
    right_leg_pos = b2Vec2(pos[0], pos[1] + torso_half_size + LEG_HALF_SIZE)
    right_leg_body = create_segment_box2d(world, right_leg_pos, 5, LEG_HALF_SIZE)
    bodys["right_leg"] = right_leg_body
    # 右髋关节 (注意：这里直接使用 HIP_MIN/MAX，但Box2D的旋转是统一的，限制需要反映其朝向)
    create_revolute_joint(
        torso_body, right_leg_body, HIP_ANCHOR, b2Vec2(0, -LEG_HALF_SIZE), 
        HIP_MIN, HIP_MAX, "right_leg"
    )
    
    # 右小腿
    right_foreleg_pos = b2Vec2(pos[0], right_leg_pos.y + LEG_HALF_SIZE + LEG_HALF_SIZE)
    right_foreleg_body = create_segment_box2d(world, right_foreleg_pos, 4, LEG_HALF_SIZE)
    bodys["right_foreleg"] = right_foreleg_body
    # 右膝关节
    create_revolute_joint(
        right_leg_body, right_foreleg_body, KNEE_ANCHOR_LEG, KNEE_ANCHOR_FORELEG, 
        KNEE_MIN, KNEE_MAX, "right_foreleg"
    )

    # --- 手臂 ---
    # 为了保持动作空间的维度 (9)，我们仍然添加电机，但不限制角度
    SHOULDER_ANCHOR = b2Vec2(0, -torso_half_size)

    # 左大臂
    left_arm_body = create_segment_box2d(world, b2Vec2(pos[0], pos[1] - torso_half_size - ARM_HALF_SIZE), 3, ARM_HALF_SIZE)
    bodys["left_arm"] = left_arm_body
    create_revolute_joint(torso_body, left_arm_body, SHOULDER_ANCHOR, b2Vec2(0, ARM_HALF_SIZE), name="left_arm")

    # 左小臂
    left_forearm_body = create_segment_box2d(world, b2Vec2(pos[0], pos[1] - torso_half_size - ARM_HALF_SIZE*3), 3, ARM_HALF_SIZE)
    bodys["left_forearm"] = left_forearm_body
    create_revolute_joint(left_arm_body, left_forearm_body, b2Vec2(0, -ARM_HALF_SIZE), b2Vec2(0, ARM_HALF_SIZE), name="left_forearm")

    # 右大臂
    right_arm_body = create_segment_box2d(world, b2Vec2(pos[0], pos[1] - torso_half_size - ARM_HALF_SIZE), 3, ARM_HALF_SIZE)
    bodys["right_arm"] = right_arm_body
    create_revolute_joint(torso_body, right_arm_body, SHOULDER_ANCHOR, b2Vec2(0, ARM_HALF_SIZE), name="right_arm")

    # 右小臂
    right_forearm_body = create_segment_box2d(world, b2Vec2(pos[0], pos[1] - torso_half_size - ARM_HALF_SIZE*3), 3, ARM_HALF_SIZE)
    bodys["right_forearm"] = right_forearm_body
    create_revolute_joint(right_arm_body, right_forearm_body, b2Vec2(0, -ARM_HALF_SIZE), b2Vec2(0, ARM_HALF_SIZE), name="right_forearm")

    return motors, bodys

# --- env runtime ---
def set_motor_torque(motors, torques):
    for name, torque in torques.items():
        joint = motors[name]
        
        joint.maxMotorTorque = float(np.clip(np.abs(torque) * MAX_TORQUE, 0, 1))
        joint.motorSpeed = float(np.sign(torque) * TARGET_SPEED)

def set_motor_rate(motors, rates):
    for name, rate in rates.items():
        motors[name].motorSpeed = float(rate * TARGET_SPEED)
    

def get_states(joints, bodys):
    state = dict()
    # 1. 身体状态 (躯干位置和角度，线性和角速度)
    torso = bodys["torso"]
    state["torso"] = {
        "pos": [torso.position[0] / GROUND_Y, torso.position[1] / GROUND_Y],
        "angle": torso.angle / (2 * b2_pi),
        "velocity": [torso.linearVelocity[0], torso.linearVelocity[1]],
        "angular_velocity": torso.angularVelocity
    }

    # 2. 关节状态 (角度, 角速度, 接触)
    for name, joint in joints.items():
        state[name] = {
            "angle": joint.angle / (2 * b2_pi),
            "velocity": joint.speed
        }

    # 3. 接触检测（手, 脚）
    for name, body in bodys.items():
        if name in ['left_forearm', 'right_forearm', 'left_foreleg', 'right_foreleg']:
            state[name]["ground"] = (GROUND_Y - body.position[1] < 2)
    
    return state

def detect_contact(world, body_a, body_b):
    for contact in world.contacts:
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        is_contact = (fixture_a.body == body_a and fixture_b.body == body_b) or \
                        (fixture_b.body == body_a and fixture_a.body == body_b)

        if is_contact:
            if contact.touching: 
                return True
    return False

# --- render util ---
PPM = 1.0 / SCALE  # 像素每米
DRAW_WITDH = 3
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

def to_screen_coords(pos_x, pos_y):
    # 转换 Y 轴方向 (Box2D Y向上，Pygame Y向下)
    screen_x = int(pos_x * PPM)
    screen_y = int((pos_y * PPM))
    return (screen_x, screen_y)

def draw_line_segment(screen, body, half_length, color):
    local_start = b2Vec2(0, half_length)
    local_end = b2Vec2(0, -half_length)

    world_start = body.GetWorldPoint(local_start)
    world_end = body.GetWorldPoint(local_end)
    
    screen_start = to_screen_coords(world_start.x, world_start.y)
    screen_end = to_screen_coords(world_end.x, world_end.y)
    
    pygame.draw.line(screen, color, screen_start, screen_end, DRAW_WITDH)

def draw_circle(screen, body, fixture, color):
    shape = fixture.shape
    center_world = body.transform * shape.pos
    radius_pixels = int(shape.radius * PPM)
    
    screen_center = to_screen_coords(center_world.x, center_world.y)
    pygame.draw.circle(screen, color, screen_center, radius_pixels, 0)

def draw_polygon(screen, body, fixture, color):
    shape = fixture.shape

    vertices = [(body.transform * v) for v in shape.vertices]

    screen_vertices = [to_screen_coords(v.x, v.y) for v in vertices]
    pygame.draw.polygon(screen, color, screen_vertices, 0)

# --- env operation ---
class MatchmanEnv(gym.Env):
    VEL_ITERS = 10
    POS_ITERS = 10
    TIME_STEP = 1.0 / 60.0

    def __init__(self, rewards, draw = False):
        super().__init__()
        self.rewards = rewards
        self.draw = draw
        self._running = False
        self.world = None
        self.pygame_init = False
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._running = True
        self.world = b2World(gravity=(0, 20))
        
        # 创建地面
        self.grounds = create_boundaries(self.world, 800 * SCALE, 600 * SCALE)

        # 创建火柴人
        self.motors, self.bodys = create_matchman(self.world, (400 * SCALE, 400 * SCALE))

        initial_state = pack_state(get_states(self.motors, self.bodys))
        return initial_state, {}

    def running(self):
        return self._running

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        action_dict = unpack_action(action)
        set_motor_rate(self.motors, action_dict)

        self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)

        if self.draw:
            self.render()

        next_state = get_states(self.motors, self.bodys)
        
        reward = np.array(sum([r(next_state, action_dict) for r in self.rewards]))
        done = detect_contact(self.world, self.bodys['head'], self.grounds[0])
        if done: 
            reward = -50

        return pack_state(next_state), reward, done, False, {}

    def render(self):
        if not self.pygame_init:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            self.pygame_init = True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                pygame.quit()
        
        self.screen.fill((255, 255, 255))

        color = (0, 0, 0)
        for ground in self.grounds:
            for fixture in ground.fixtures:
                draw_polygon(self.screen, ground, fixture, color)
        for name, body in self.bodys.items():
            if name == "head":
                for fixture in body.fixtures:
                    draw_circle(self.screen, body, fixture, color)
            else:
                half_size = limb_half_size.get(name)
                if half_size is not None:
                    draw_line_segment(self.screen, body, half_size, color)
        
        pygame.display.flip()
        self.clock.tick(1.0 / self.TIME_STEP)

    def close(self):
        pygame.quit()
