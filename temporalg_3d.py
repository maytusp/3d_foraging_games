#TODO Add location as observation
#TODO Add PettingZoo Wrapper

import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import math
import cv2

import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

class TemporalGEnv:
    def __init__(self, headless=False, image_size=96):
        self.mode = p.DIRECT if headless else p.GUI
        self.headless = headless
        # self.client = p.connect(self.mode)
        self.client = None
        
        self.dt = 1./100.
        self.ROBOT_SCALE = 0.6 

        self.FORWARD_DIST = 0.25
        self.TARGET_ANGLE = 20
        self.ACTION_DURATION = 0.1 
        self.LIN_SPEED = self.FORWARD_DIST / self.ACTION_DURATION
        self.ANG_SPEED = math.radians(self.TARGET_ANGLE) / self.ACTION_DURATION
        self.STEPS_PER_ACTION = int(self.ACTION_DURATION / self.dt)

        self.FREEZE_STEPS = 6
        self.TARGET_SIZE = 0.3
        self.COLLECT_DIST = 0.5
        self.ARENA_WIDTH = 6
        self.ARENA_HEIGHT = 6

        self.IMAGE_SIZE = image_size

        # --- NEW CONSTANTS FOR COMMUNICATION ---
        self.COMM_DIST_LIMIT = 2      # Meters
        self.COMM_MAX_STEPS = 5        # Steps allowed after first contact
        
        self.agent0 = None
        self.agent1 = None
        self.door_id = None
        self.middle_wall_ids = []

    def _setup_pybullet(self):
        """Initialise PyBullet"""
        self.mode = p.DIRECT if self.headless else p.GUI
        self.client = p.connect(self.mode)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.8)
        
        # Load Static Assets
        p.loadURDF("plane.urdf")
        self._create_arena_boundary(self.ARENA_WIDTH, self.ARENA_HEIGHT)
        
        # Load Agents
        self.agent0 = self._load_r2d2_asset([0, -2, -5])
        self.agent1 = self._load_r2d2_asset([0, 2, -5])
        self.agents = [self.agent0, self.agent1]
        
        # Load Items
        self.item_red = self._create_sphere_item([0,0,-5], [1,0,0,1])
        self.item_blue = self._create_sphere_item([0,0,-5], [0,0,1,1])
        self.items = [self.item_red, self.item_blue]

    def reset(self):
        reconnect = False
        if self.client is None:
            reconnect = True
        else:
            try:
                p.getConnectionInfo(self.client)
            except:
                reconnect = True
        
        if reconnect:
            self._setup_pybullet()
        # Clean up old door AND old middle walls
        if self.door_id is not None:
            p.removeBody(self.door_id)
            self.door_id = None
        
        for body_id in self.middle_wall_ids:
            p.removeBody(body_id)
        self.middle_wall_ids = [] 

        # Respawn Door & Middle Wall
        self._create_random_door_and_walls(self.ARENA_WIDTH)

        # Teleport Agents
        rand_x0, rand_y0 = random.uniform(-2.3, 2.3), random.uniform(-2.3, -1.2)
        self._teleport_agent(self.agent0, [rand_x0, rand_y0], yaw=1.57)
        
        rand_x1, rand_y1 = random.uniform(-2.3, 2.3), random.uniform(1.2, 2.3)
        self._teleport_agent(self.agent1, [rand_x1, rand_y1], yaw=-1.57)

        # Reset Items
        self._reset_item_positions_in_front(dist=0.6)
        
        self.step_count = 0
        self._setup_temporal_logic()
        
        # --- RESET COMMUNICATION STATE ---
        self.has_met = False        # Have they ever met?
        self.comm_timer = 0         # How long since they first met?
        self.comm_mask = 0          # Current step mask (1=Talk, 0=Silent)

        return self._get_obs()

    def _create_arena_boundary(self, width, height, wall_thickness=0.2, wall_height=0.7):
        horiz_half = [width / 2 + wall_thickness, wall_thickness / 2, wall_height / 2]
        vert_half = [wall_thickness / 2, height / 2 + wall_thickness, wall_height / 2]
        
        wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=horiz_half)
        wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=horiz_half, rgbaColor=[0.7, 0.7, 0.7, 1])
        p.createMultiBody(0, wall_col, wall_vis, [0, height/2, wall_height/2])
        p.createMultiBody(0, wall_col, wall_vis, [0, -height/2, wall_height/2])
        
        wall_col_v = p.createCollisionShape(p.GEOM_BOX, halfExtents=vert_half)
        wall_vis_v = p.createVisualShape(p.GEOM_BOX, halfExtents=vert_half, rgbaColor=[0.7, 0.7, 0.7, 1])
        p.createMultiBody(0, wall_col_v, wall_vis_v, [width/2, 0, wall_height/2])
        p.createMultiBody(0, wall_col_v, wall_vis_v, [-width/2, 0, wall_height/2])
        
        self.wall_thickness = wall_thickness
        self.wall_height = wall_height

    def _create_random_door_and_walls(self, width):
        '''
        Easiest Scenario is the long middle wall is a door that can disappear after being hit by agents
        '''
        door_w = 6.0
        door_x = 0
        door_half = [door_w/2, self.wall_thickness/2, self.wall_height/2]
        self.door_id = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=door_half), 
                                        p.createVisualShape(p.GEOM_BOX, halfExtents=door_half, rgbaColor=[0.54, 0.27, 0.07, 1]), 
                                        [door_x, 0, self.wall_height/2])


    

    # def _create_random_door_and_walls(self, width):
    '''
    TODO Increase difficulty as option.
    This function is for the random door placed on the middle wall.
    Level 1: The entire middle wall is a door.
    Level 2: The small door is randomly placed on the middle wall.
    Level 3: Obstacles like trees.
    Level 4: Uneven terrains.
    '''
    #     door_w = 1.0
    #     door_x = random.uniform(-(width/2 - door_w), (width/2 - door_w))
    #     door_half = [door_w/2, self.wall_thickness/2, self.wall_height/2]
    #     self.door_id = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=door_half), 
    #                                     p.createVisualShape(p.GEOM_BOX, halfExtents=door_half, rgbaColor=[0.54, 0.27, 0.07, 1]), 
    #                                     [door_x, 0, self.wall_height/2])

    #     l_width = (door_x - door_w/2) - (-width/2)
    #     if l_width > 0.05:
    #         h = [l_width/2, self.wall_thickness/2, self.wall_height/2]
    #         pos_x = -width/2 + l_width/2
    #         wall_id = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=h),
    #                                    p.createVisualShape(p.GEOM_BOX, halfExtents=h, rgbaColor=[0.5, 0.5, 0.5, 1]),
    #                                    [pos_x, 0, self.wall_height/2])
    #         self.middle_wall_ids.append(wall_id)

    #     r_width = (width/2) - (door_x + door_w/2)
    #     if r_width > 0.05:
    #         h = [r_width/2, self.wall_thickness/2, self.wall_height/2]
    #         pos_x = width/2 - r_width/2
    #         wall_id = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=h),
    #                                    p.createVisualShape(p.GEOM_BOX, halfExtents=h, rgbaColor=[0.5, 0.5, 0.5, 1]),
    #                                    [pos_x, 0, self.wall_height/2])
    #         self.middle_wall_ids.append(wall_id)

    def _update_communication_logic(self):
        # 1. Calculate Distance
        pos0, _ = p.getBasePositionAndOrientation(self.agent0)
        pos1, _ = p.getBasePositionAndOrientation(self.agent1)
        dist = math.sqrt((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)
        
        # 2. Check "First Met" Trigger
        if dist <= self.COMM_DIST_LIMIT and not self.has_met:
            self.has_met = True
            # print(f">>> [Step {self.step_count}] Agents First Met! Timer Started.")

        # 3. Update Timer (ticks only if they have met at least once)
        if self.has_met:
            self.comm_timer += 1

        # 4. Determine Mask
        # Condition A: Must be close NOW
        # Condition B: Must have time remaining on the clock
        is_close_now = (dist <= self.COMM_DIST_LIMIT)
        time_remaining = (self.comm_timer <= self.COMM_MAX_STEPS) # inclusive or exclusive depending on preference
        
        if self.has_met and time_remaining:
            self.comm_mask = 1
        else:
            self.comm_mask = 0
            
        return self.comm_mask

    def step(self, actions):
        if self.step_count < self.FREEZE_STEPS:
            actions = [0, 0]
        self.step_count += 1
        
        for _ in range(self.STEPS_PER_ACTION):
            for i, agent_id in enumerate(self.agents):
                self._apply_velocity_momentarily(agent_id, actions[i])
            p.stepSimulation()
            
            if self.door_id is not None:
                for agent in self.agents:
                    contacts = p.getContactPoints(bodyA=agent, bodyB=self.door_id)
                    if contacts:
                        p.removeBody(self.door_id)
                        self.door_id = None
                        break

        for agent_id in self.agents:
            p.resetBaseVelocity(agent_id, [0,0,0], [0,0,0])

        self._check_spawns()
        
        # --- UPDATE COMMUNICATION MASK ---
        comm_mask = self._update_communication_logic()

        img_obs, loc_obs = self._get_obs()
        
        d_target = self._get_dists(self.items[self.target_idx])
        d_distractor = self._get_dists(self.items[self.distractor_idx])
        
        reward, done, status = 0, False, ""
        if d_target[0] < self.COLLECT_DIST and d_target[1] < self.COLLECT_DIST:
            reward = 1.0; done = True; status = "SUCCESS"
        elif d_distractor[0] < self.COLLECT_DIST and d_distractor[1] < self.COLLECT_DIST:
            reward = 0.1; done = True; status = "DISTRACTOR"

        # Note: Added comm_mask to return values
        return img_obs, loc_obs, reward, done, status, comm_mask

    def _apply_velocity_momentarily(self, agent_id, action):
        lin_v = self.LIN_SPEED if action == 1 else 0
        ang_v = self.ANG_SPEED if action == 2 else (-self.ANG_SPEED if action == 3 else 0)
        _, orn = p.getBasePositionAndOrientation(agent_id)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        global_lin = rot_mat.dot([0, lin_v, 0])
        p.resetBaseVelocity(agent_id, global_lin, [0, 0, ang_v])

    def _reset_item_positions_in_front(self, dist):
        def _get_item_target_pos(agent, d):
            p_pos, p_orn = p.getBasePositionAndOrientation(agent)
            r = np.array(p.getMatrixFromQuaternion(p_orn)).reshape(3,3)
            return np.array(p_pos) + r.dot([0, d, 0])

        pos0 = _get_item_target_pos(self.agent0, dist)
        pos1 = _get_item_target_pos(self.agent1, dist)
        
        self._teleport_item(self.item_red, [pos0[0], pos0[1], -5.0])
        self._teleport_item(self.item_blue, [pos1[0], pos1[1], -5.0])

    def _setup_temporal_logic(self):
        times = sorted(random.sample(range(1, self.FREEZE_STEPS + 1), 2))
        self.target_idx = random.choice([0, 1])
        self.distractor_idx = 1 - self.target_idx
        self.spawn_time_target, self.spawn_time_distractor = times
        self.target_spawned = self.distractor_spawned = False
        # print(f"Goal: {'RED' if self.target_idx == 0 else 'BLUE'}")

    def _check_spawns(self):
        if self.step_count == self.spawn_time_target and not self.target_spawned:
            self._pop_item_up(self.items[self.target_idx]); self.target_spawned = True  
        if self.step_count == self.spawn_time_distractor and not self.distractor_spawned:
            self._pop_item_up(self.items[self.distractor_idx]); self.distractor_spawned = True

    def _load_r2d2_asset(self, pos):
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        agent_id = p.loadURDF("r2d2.urdf", pos, start_orn, globalScaling=self.ROBOT_SCALE)
        for j in range(p.getNumJoints(agent_id)):
            p.changeDynamics(agent_id, j, lateralFriction=0.0, spinningFriction=0.0)
        p.changeDynamics(agent_id, -1, lateralFriction=0.0, spinningFriction=0.0)
        return agent_id
    
    def _teleport_agent(self, agent_id, pos, yaw):
        new_pos = [pos[0], pos[1], 0.1]
        new_orn = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(agent_id, new_pos, new_orn)

    def _create_sphere_item(self, pos, color):
        v = p.createVisualShape(p.GEOM_SPHERE, radius=self.TARGET_SIZE, rgbaColor=color)
        return p.createMultiBody(0, -1, v, basePosition=pos)
        
    def _teleport_item(self, item_id, pos):
        _, orn = p.getBasePositionAndOrientation(item_id)
        p.resetBasePositionAndOrientation(item_id, pos, orn)
        
    def _pop_item_up(self, item_id):
        pos, orn = p.getBasePositionAndOrientation(item_id)
        p.resetBasePositionAndOrientation(item_id, [pos[0], pos[1], 0.2], orn)

    def _get_dists(self, item_id):
        i_pos, _ = p.getBasePositionAndOrientation(item_id)
        return [math.sqrt((p.getBasePositionAndOrientation(a)[0][0]-i_pos[0])**2 + 
                (p.getBasePositionAndOrientation(a)[0][1]-i_pos[1])**2) for a in self.agents]

    def _get_obs(self):
        img_obs = []
        loc_obs = []
        
        for agent_id, agent in enumerate(self.agents):
            pos, orn = p.getBasePositionAndOrientation(agent)
            r = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
            
            x, y = pos[0], pos[1]
            forward_vec = r.dot(np.array([0, 1, 0])) 
            dx, dy = forward_vec[0], forward_vec[1]
            loc_obs.append(np.array([x, y, dx, dy], dtype=np.float32))
            
            # --- 3. Vision ---
            offset_eye = np.array([0, -0.2, 0.7]) * self.ROBOT_SCALE
            offset_target = np.array([0, 1.0, 0.7]) * self.ROBOT_SCALE
            
            cam_eye = np.array(pos) + r.dot(offset_eye)
            cam_target = np.array(pos) + r.dot(offset_target)
            
            view = p.computeViewMatrix(cam_eye, cam_target, [0,0,1])
            proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
            _, _, rgb, _, _ = p.getCameraImage(self.IMAGE_SIZE, self.IMAGE_SIZE, view, proj, renderer=p.ER_TINY_RENDERER)
            
            # Helper: Resize here if you need 224x224 immediately, otherwise keep 256
            rgb_img = cv2.cvtColor(np.array(rgb, dtype=np.uint8).reshape(self.IMAGE_SIZE, self.IMAGE_SIZE, 4)[:, :, :3], cv2.COLOR_RGB2BGR)
            img_obs.append(rgb_img)
        # print(f"agent 0 is at (x,y,dx,dy) = {loc_obs[0]}")
        # print(f"agent 1 is at (x,y,dx,dy) = {loc_obs[1]}")
        return img_obs, loc_obs


class PettingZooWrapper(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "3d_temporalg_v1"}

    def __init__(self, headless=True, image_size=96):
        self.render_mode = None if headless else "human"
        self.possible_agents = ["agent_0", "agent_1"]
        self.env = TemporalGEnv(headless=headless, image_size=image_size)
        
        # Simple Actions: Move Forward, Turn Left, Turn Right
        self.action_space_map = {agent: spaces.Discrete(3) for agent in self.possible_agents}
        self.observation_space_map = {
            agent: spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(3, self.env.IMAGE_SIZE, self.env.IMAGE_SIZE), dtype=np.uint8),
                "location": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "mask": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32)
            }) for agent in self.possible_agents
        }
        
        self.comm_mask = 0

    @property
    def observation_space(self):
        return lambda agent: self.observation_space_map[agent]

    @property
    def action_space(self):
        return lambda agent: self.action_space_map[agent]

    def _process_obs(self, img_obs_list, loc_obs_list):
        observations = {}
        for i, agent in enumerate(self.agents):
            raw_img = img_obs_list[i]
            resized_img = cv2.resize(raw_img, (self.env.IMAGE_SIZE, self.env.IMAGE_SIZE))
            image_tensor = np.transpose(resized_img, (2, 0, 1))

            observations[agent] = {
                "image": image_tensor,
                "location": loc_obs_list[i],
                "mask": np.array([self.comm_mask], dtype=np.int32),
            }
        return observations

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Env reset returns (img_obs, loc_obs)
        img_obs, loc_obs = self.env.reset()
        
        self.agents = self.possible_agents[:]
        self.comm_mask = self.env.comm_mask
        observations = self._process_obs(img_obs, loc_obs)
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos


    def step(self, actions):
        if isinstance(actions, dict):
            # Map agent names to their actions strictly in order
            actions = [actions[a] for a in self.possible_agents]
        elif isinstance(actions, list) or isinstance(actions, np.ndarray):
            pass
        else:
            raise TypeError(f"Expected dict or list actions, got {type(actions)}")
        img_obs, loc_obs, reward, done, status, comm_mask = self.env.step(actions)
        self.comm_mask = comm_mask

        observations = self._process_obs(img_obs, loc_obs)
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"status": status} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos

# if __name__ == "__main__":
#     env = TemporalGEnv(headless=False)
#     env.reset()
    
#     print("------------------------------------------------")
#     print("Controls: Arrows (Agent 0), I/J/L (Agent 1)")
#     print("R: Reset | ESC: Quit")
#     print("------------------------------------------------")

#     while True:
#         keys = p.getKeyboardEvents()
#         a0, a1, step_triggered = 0, 0, False
        
#         def is_active(k):
#             return k in keys and (keys[k] & p.KEY_IS_DOWN or keys[k] & p.KEY_WAS_TRIGGERED)

#         if is_active(p.B3G_UP_ARROW): a0 = 1; step_triggered = True
#         if is_active(p.B3G_LEFT_ARROW): a0 = 2; step_triggered = True
#         if is_active(p.B3G_RIGHT_ARROW): a0 = 3; step_triggered = True
        
#         if is_active(ord('i')): a1 = 1; step_triggered = True
#         if is_active(ord('j')): a1 = 2; step_triggered = True
#         if is_active(ord('l')): a1 = 3; step_triggered = True
        
#         if is_active(ord('r')): 
#             env.reset()
#             step_triggered = False
        
#         if is_active(27): 
#             break
        
#         if step_triggered:
#             img_obs, loc_obs, reward, done, status, comm_mask = env.step([a0, a1])
            
#             # --- DEBUG VISUALIZATION ---
#             # Print status to console so you can verify logic
#             if comm_mask:
#                 print(f"COMM ACTIVE | Steps left: {env.COMM_MAX_STEPS - env.comm_timer}")
#             elif env.has_met and env.comm_timer > env.COMM_MAX_STEPS:
#                 print("COMM EXPIRED (Time up)")
#             elif not env.has_met:
#                 print("COMM INACTIVE (Haven't met)")
#             else:
#                 print("COMM BLOCKED (Too far)")
#             # ---------------------------

#             cv2.imshow("Views", np.hstack((img_obs[0], img_obs[1])))
#             if done:
#                 print(f">>> {status}"); cv2.waitKey(1500); env.reset()
        
#         cv2.waitKey(1)


if __name__ == "__main__":
    # Initialize the PettingZoo Wrapper instead of raw Env
    env = PettingZooWrapper(headless=False)
    env.reset()
    
    print("------------------------------------------------")
    print("Controls: Arrows (Agent 0), I/J/L (Agent 1)")
    print("R: Reset | ESC: Quit")
    print("------------------------------------------------")

    while True:
        keys = p.getKeyboardEvents()
        a0, a1, step_triggered = 0, 0, False
        
        def is_active(k):
            return k in keys and (keys[k] & p.KEY_IS_DOWN or keys[k] & p.KEY_WAS_TRIGGERED)

        # Agent 0 Controls
        if is_active(p.B3G_UP_ARROW): a0 = 1; step_triggered = True
        if is_active(p.B3G_LEFT_ARROW): a0 = 2; step_triggered = True
        if is_active(p.B3G_RIGHT_ARROW): a0 = 3; step_triggered = True
        
        # Agent 1 Controls
        if is_active(ord('i')): a1 = 1; step_triggered = True
        if is_active(ord('j')): a1 = 2; step_triggered = True
        if is_active(ord('l')): a1 = 3; step_triggered = True
        
        if is_active(ord('r')): 
            env.reset()
            step_triggered = False
            print(">>> RESET")
        
        if is_active(27): # ESC
            break
        
        if step_triggered:
            # 1. Construct PettingZoo Action Dictionary
            actions = [a0, a1]
            
            # 2. Step the Wrapper
            observations, rewards, terms, truncs, infos = env.step(actions)
            
            # 3. Extract Data for Visualization
            # Wrapper returns (3, H, W) for PyTorch -> Transpose to (H, W, 3) for OpenCV
            vis0 = np.transpose(observations["agent_0"]["image"], (1, 2, 0))
            vis1 = np.transpose(observations["agent_1"]["image"], (1, 2, 0))
            
            # Extract Mask (stored in observation)
            comm_mask = observations["agent_0"]["mask"][0]
            
            # --- DEBUG VISUALIZATION ---
            # Access internal env variables just for debug printing
            inner_env = env.env 
            
            if comm_mask:
                print(f"COMM ACTIVE | Steps left: {inner_env.COMM_MAX_STEPS - inner_env.comm_timer}")
            elif inner_env.has_met and inner_env.comm_timer > inner_env.COMM_MAX_STEPS:
                print("COMM EXPIRED (Time up)")
            elif not inner_env.has_met:
                print("COMM INACTIVE (Haven't met)")
            else:
                print("COMM BLOCKED (Too far)")
            # ---------------------------

            # Show Views
            cv2.imshow("Views (Agent 0 | Agent 1)", np.hstack((vis0, vis1)))
            
            # Check Done (Any agent done = episode done in this wrapper)
            if any(terms.values()) or any(truncs.values()):
                status = infos["agent_0"].get("status", "DONE")
                print(f">>> {status}")
                cv2.waitKey(1500)
                env.reset()
        
        cv2.waitKey(1)