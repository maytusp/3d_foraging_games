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

import pkgutil

class TemporalGEnv:
    def __init__(self, headless, image_size, max_steps, collect_distractor, eval):
        self.collect_distractor = collect_distractor
        self.distractor_reward = False # The reward is for initialize navigation behaviour to go to object
        self.partial_reward = False
        self.mode = p.DIRECT if headless else p.GUI
        self.headless = headless
        self.max_steps = max_steps  # Store the limit
        # self.client = p.connect(self.mode)
        self.client = None
        
        self.dt = 1./30.  # Faster physics tick
        self.ACTION_DURATION = 0.1 # Slightly longer action
        self.ROBOT_SCALE = 0.3

        self.FORWARD_DIST = 0.5
        self.TARGET_ANGLE = 45
        self.LIN_SPEED = self.FORWARD_DIST / self.ACTION_DURATION
        self.ANG_SPEED = math.radians(self.TARGET_ANGLE) / self.ACTION_DURATION
        self.STEPS_PER_ACTION = int(self.ACTION_DURATION / self.dt)
        self.eval = eval
        self.FREEZE_STEPS = 6
        self.TARGET_SIZE = 0.3
        self.COLLECT_DIST = 1.0
        self.ARENA_WIDTH = 6
        self.ARENA_HEIGHT = 6

        self.IMAGE_SIZE = image_size

        self.COMM_MAX_STEPS = 5        # Message length
        
        self.agent0 = None
        self.agent1 = None
        self.door_id = None
        self.middle_wall_ids = []

        self.SPAWN_FLASH_STEPS = 1   # how many env-steps the cue lasts
        self.flash_remaining = [0, 0]  # per-agent countdown
        self.flash_tint = [
            np.array([60, 0, 0], dtype=np.float32),   # agent0 sees item0 -> red tint
            np.array([60, 0, 0], dtype=np.float32),   # agent1 sees item1 -> blue tint
        ]
        self.flash_gain = 3  # brightness multiplier

    def _setup_pybullet(self):
        """Initialise PyBullet"""
        self.mode = p.DIRECT if self.headless else p.GUI
        self.client = p.connect(self.mode)

        # if self.headless:
        #     try:
        #         egl = pkgutil.get_loader('eglRenderer')
        #         if egl:
        #             plugin_id = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        #         else:
        #             plugin_id = p.loadPlugin("eglRendererPlugin")
                
        #         if plugin_id >= 0:
        #             print(f"GPU Renderer Loaded (ID: {plugin_id})")
        #         else:
        #             print("GPU Renderer Failed to Load (Using CPU fallback)")
        #     except Exception as e:
        #         print(f"EGL Load Error: {e}")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.8)
        
        # Load Static Assets
        p.loadURDF("plane.urdf")
        self._create_arena_boundary(self.ARENA_WIDTH, self.ARENA_HEIGHT)
        
        # Load Agents
        self.agent0 = self._load_faced_agent([0, -2, -5], [0, 1, 1, 1]) 
        self.agent1 = self._load_faced_agent([0, 2, -5], [1, 0, 1, 1])
        self.agents = [self.agent0, self.agent1]
        
        # Load Items
        self.item_red = self._create_sphere_item([0,0,-5], [1,0,0,1])
        self.item_blue = self._create_sphere_item([0,0,-5], [1,0,0,1])
        self.items = [self.item_red, self.item_blue]



    def reset(self):
        self.flash_remaining = [0, 0]
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
        rand_x0, rand_y0 = random.uniform(-2, 2), random.uniform(-2.7, -2)
        self._teleport_agent(self.agent0, [rand_x0, rand_y0], yaw=1.57)
        
        rand_x1, rand_y1 = random.uniform(-2, 2), random.uniform(2.7, 2)
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
        door_w = self.ARENA_WIDTH
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
        pos0, _ = p.getBasePositionAndOrientation(self.agent0)
        pos1, _ = p.getBasePositionAndOrientation(self.agent1)
        dist = math.sqrt((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)
        
        if self.has_met:
            self.comm_timer += 1

        time_remaining = (self.comm_timer <= self.COMM_MAX_STEPS) # inclusive or exclusive depending on preference
        
        if self.has_met and time_remaining:
            self.comm_mask = 1
        else:
            self.comm_mask = 0
            
        return self.comm_mask

    def render_global(self):
        """
        Renders a 3rd person 'God View' of the arena.
        """
        # 1. Camera Position: High up (z=4), looking down at center (0,0,0)
        # You can adjust pitch/yaw/distance to change the angle
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=5.0,
            yaw=0,
            pitch=-90,  # -90 is Top-Down, -60 is Isometric
            roll=0,
            upAxisIndex=2
        )
        
        # 2. Projection Matrix (Field of View)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )
        
        # 3. Render
        # We use a higher resolution (e.g., 480x480) for the video than the agent sees
        width, height = 480, 480
        _, _, px, _, _ = p.getCameraImage(
            width=width, 
            height=height, 
            viewMatrix=view_matrix, 
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER # Works on CPU/Headless
        )
        
        # 4. Process Image (RGBA -> RGB)
        rgb_array = np.array(px, dtype=np.uint8).reshape(height, width, 4)
        rgb_array = rgb_array[:, :, :3] # Drop Alpha channel
        return rgb_array

    def step(self, actions):
        for i in range(2):
            if self.flash_remaining[i] > 0:
                self.flash_remaining[i] -= 1
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
                        self.has_met = True
                        break

        for agent_id in self.agents:
            p.resetBaseVelocity(agent_id, [0,0,0], [0,0,0])

        self._check_spawns()

        # --- UPDATE COMMUNICATION MASK ---
        comm_mask = self._update_communication_logic()

        img_obs, loc_obs, spawn_time_obs, time_step_obs = self._get_obs()
        
        d_target = self._get_dists(self.items[self.target_idx])
        d_distractor = self._get_dists(self.items[self.distractor_idx])
        
        reward, done, status = 0.0, False, ""

        # TERMINAL REWARD
        if d_target[0] < self.COLLECT_DIST and d_target[1] < self.COLLECT_DIST:
            reward = 1.0; done = True; status = "SUCCESS"
        # IF COLLECT DISTRACTOR
        elif self.collect_distractor and d_distractor[0] < self.COLLECT_DIST and d_distractor[1] < self.COLLECT_DIST:
            done = True; status = "DISTRACTOR"
            if self.distractor_reward:
                reward = 0.05
        # TERMINAL PARTIAL REWARD
        elif self.step_count >= self.max_steps:
            done = True
            status = "TIMEOUT"
            if self.partial_reward:
                ZONE_OUTER = 1.5
                ZONE_INNER = 1.0
                
                # Agent 0 Score
                if d_target[0] < ZONE_INNER and d_target[1] < ZONE_INNER:
                    reward = 0.20
                elif d_target[0] < ZONE_OUTER and d_target[1] < ZONE_OUTER:
                    reward = 0.10

        # Note: Added comm_mask to return values
        return img_obs, loc_obs, spawn_time_obs, time_step_obs, reward, done, status, comm_mask

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
        if self.eval: # remove first and last time steps seen during training
            times = sorted(random.sample(range(2, self.FREEZE_STEPS), 2))
        else:
            times = sorted(random.sample(range(1, self.FREEZE_STEPS + 1), 2))

        self.target_idx = random.choice([0, 1])
        self.distractor_idx = 1 - self.target_idx
        self.spawn_time_target, self.spawn_time_distractor = times
        self.target_spawned = self.distractor_spawned = False

    def _get_time_labels(self):
        """
        Returns a list [time_for_agent_0, time_for_agent_1]
        Logic: Item 0 is always in front of Agent 0. Item 1 is always in front of Agent 1.
        """
        # If Item 0 is the target, Agent 0 gets target time.
        if self.target_idx == 0:
            t0 = self.spawn_time_target
            t1 = self.spawn_time_distractor
        # If Item 1 is the target (Item 0 is distractor), Agent 0 gets distractor time.
        else:
            t0 = self.spawn_time_distractor
            t1 = self.spawn_time_target
            
        return [t0, t1]
    
    def _get_time_steps(self):
        return [self.step_count] * 2
        
    def _check_spawns(self):
        spawned_item_idx = None

        if self.step_count == self.spawn_time_target and not self.target_spawned:
            spawned_item_idx = self.target_idx
            self._pop_item_up(self.items[self.target_idx])
            self.target_spawned = True

        if self.step_count == self.spawn_time_distractor and not self.distractor_spawned:
            spawned_item_idx = self.distractor_idx
            self._pop_item_up(self.items[self.distractor_idx])
            self.distractor_spawned = True

        if spawned_item_idx is not None:
            self.flash_remaining[spawned_item_idx] = self.SPAWN_FLASH_STEPS


    def _load_faced_agent(self, pos, eye_color):
        """
        Creates a single Physics Body with a 'Face'.
        Base: Grey Box (Collidable)
        Link: Colored 'Eye' Box (Visual only, Fixed to Base)
        """
        # 1. Main Body Params
        body_half = [0.2, 0.2, 0.3] 
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half)
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=body_half, rgbaColor=[0.2, 0.2, 0.2, 1])

        # 2. Eye Params (The front face)
        # Sits slightly in front (y=0.2) and up
        eye_half = [0.02, 0.2, 0.02] 
        eye_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=eye_half, rgbaColor=eye_color)

        # 3. Create One MultiBody
        # Note: We set linkCollisionShapeIndices to -1 so the eye does not affect physics
        agent_id = p.createMultiBody(
            baseMass=100,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=pos,
            
            linkMasses=[0.01],
            linkCollisionShapeIndices=[-1], # No collision for eye
            linkVisualShapeIndices=[eye_vis],
            linkPositions=[[0, 0.22, 0.3]], # 0.2 (body) + 0.02 (eye) = 0.22 forward
            linkOrientations=[[0,0,0,1]],
            linkInertialFramePositions=[[0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0,0,0]]
        )
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
        time_labels = self._get_time_labels()
        time_steps = self._get_time_steps()
        for agent_id, agent in enumerate(self.agents):
            pos, orn = p.getBasePositionAndOrientation(agent)
            r = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
            
            x, y = pos[0], pos[1]
            forward_vec = r.dot(np.array([0, 1, 0])) 
            dx, dy = forward_vec[0], forward_vec[1]
            loc_obs.append(np.array([x, y, dx, dy], dtype=np.float32))
            
            # --- 3. Vision ---
            offset_eye = np.array([0, 0.0, 0.05]) * self.ROBOT_SCALE
            offset_target = np.array([0, 1.0, 0.05]) * self.ROBOT_SCALE
            
            cam_eye = np.array(pos) + r.dot(offset_eye)
            cam_target = np.array(pos) + r.dot(offset_target)
            
            view = p.computeViewMatrix(cam_eye, cam_target, [0,0,1])
            proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
            _, _, rgb, _, _ = p.getCameraImage(
                            self.IMAGE_SIZE, 
                            self.IMAGE_SIZE, 
                            view, 
                            proj, 
                            shadow=0, # NO SHADOWS
                            lightDirection=[0,0,1], # Simple lighting
                            # renderer=p.ER_BULLET_HARDWARE_OPENGL, # GPU
                            renderer=p.ER_TINY_RENDERER, # CPU
                        )

            rgb_arr = np.array(rgb, dtype=np.uint8).reshape(self.IMAGE_SIZE, self.IMAGE_SIZE, 4)
            rgb_img = rgb_arr[:, :, :3] # Drop Alpha, keep RGB. No cv2 conversion needed if PyBullet returns RGB.
            if self.flash_remaining[agent_id] > 0:
                img_f = rgb_img.astype(np.float32)
                img_f = img_f * self.flash_gain + self.flash_tint[agent_id]
                rgb_img = np.clip(img_f, 0, 255).astype(np.uint8)

            img_obs.append(rgb_img)
        # print(f"agent 0 is at (x,y,dx,dy) = {loc_obs[0]}")
        # print(f"agent 1 is at (x,y,dx,dy) = {loc_obs[1]}")
        return img_obs, loc_obs, time_labels, time_steps


class PettingZooWrapper(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "3d_temporalg_v1"}

    def __init__(self, headless=True, image_size=46, max_steps=32, collect_distractor=True, eval=False):
        self.render_mode = None if headless else "human"
        self.possible_agents = ["agent_0", "agent_1"]
        self.env = TemporalGEnv(headless=headless, image_size=image_size, max_steps=max_steps, collect_distractor=collect_distractor, eval=eval)
        self.FREEZE_STEPS = self.env.FREEZE_STEPS
        # Simple Actions: Move Forward, Turn Left, Turn Right
        self.action_space_map = {agent: spaces.Discrete(3) for agent in self.possible_agents}
        self.observation_space_map = {
            agent: spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(3, self.env.IMAGE_SIZE, self.env.IMAGE_SIZE), dtype=np.uint8),
                "location": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "mask": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
                "spawn_time": spaces.Box(low=1, high=self.FREEZE_STEPS, shape=(1,), dtype=np.int32),
                "time_step": spaces.Box(low=1, high=1000, shape=(1,), dtype=np.int32)
            }) for agent in self.possible_agents
        }
        
        self.comm_mask = 0

    @property
    def observation_space(self):
        return lambda agent: self.observation_space_map[agent]

    @property
    def action_space(self):
        return lambda agent: self.action_space_map[agent]

    def _process_obs(self, img_obs_list, loc_obs_list, spawn_time_obs_list, time_step_obs_list):
        observations = {}
        for i, agent in enumerate(self.agents):
            raw_img = img_obs_list[i]
            resized_img = cv2.resize(raw_img, (self.env.IMAGE_SIZE, self.env.IMAGE_SIZE))
            image_tensor = np.transpose(resized_img, (2, 0, 1))

            observations[agent] = {
                "image": image_tensor,
                "location": loc_obs_list[i],
                "mask": np.array([self.comm_mask], dtype=np.int32),
                "spawn_time": np.array([spawn_time_obs_list[i]], dtype=np.int32),
                "time_step": np.array([time_step_obs_list[i]], dtype=np.int32),
            }
        return observations

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        img_obs, loc_obs, spawn_time_obs, time_step_obs = self.env.reset()
        
        self.agents = self.possible_agents[:]
        self.comm_mask = self.env.comm_mask

        self.episode_return = {a: 0.0 for a in self.agents}
        self.episode_length = {a: 0 for a in self.agents}
        
        observations = self._process_obs(img_obs, loc_obs, spawn_time_obs, time_step_obs)
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def render(self):
        return self.env.render_global()


    def step(self, actions):
        if isinstance(actions, dict):
            # Map agent names to their actions strictly in order
            actions = [actions[a] for a in self.possible_agents]
        elif isinstance(actions, list) or isinstance(actions, np.ndarray):
            pass
        else:
            raise TypeError(f"Expected dict or list actions, got {type(actions)}")
        img_obs, loc_obs, spawn_time_obs, time_step_obs, reward, done, status, comm_mask = self.env.step(actions)
        self.comm_mask = comm_mask

        observations = self._process_obs(img_obs, loc_obs, spawn_time_obs, time_step_obs)
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for i, agent in enumerate(self.agents):
            # Accumulate stats
            self.episode_return[agent] += reward
            self.episode_length[agent] += 1
            
            rewards[agent] = reward
            terminations[agent] = done
            truncations[agent] = False 
            
            infos[agent] = {"status": status}
            
            if done:
                infos[agent]["episode"] = {
                    "r": self.episode_return[agent],
                    "l": self.episode_length[agent]
                }
        
        return observations, rewards, terminations, truncations, infos


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


            # SWAP RGB -> BGR for OpenCV
            vis0_bgr = cv2.cvtColor(vis0, cv2.COLOR_RGB2BGR)
            vis1_bgr = cv2.cvtColor(vis1, cv2.COLOR_RGB2BGR)


        
            cv2.imshow("Views (Agent 0 | Agent 1)", np.hstack((vis0_bgr, vis1_bgr)))
            
            # Check Done (Any agent done = episode done in this wrapper)
            if any(terms.values()) or any(truncs.values()):
                status = infos["agent_0"].get("status", "DONE")
                print(f">>> {status}")
                cv2.waitKey(1500)
                env.reset()
        
        cv2.waitKey(1)