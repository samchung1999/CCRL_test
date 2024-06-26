import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
import gym
from gym import spaces
from copy import deepcopy
import os
import sys

abspath = os.path.dirname(os.path.abspath(__file__))


class ExploreEnv_v1(gym.Env):
    def __init__(self, map_name='train_map_l1', random_obstacle=False, training=False, render=False):
        self.explore_map_size = 24
        self.local_map_size_half = 12
        self.episode_limit = 300
        self.occupied_pixel = 255
        self.unknown_pixel = 128
        self.free_pixel = 0
        self.explored_pixel = 255
        self.agent_pixel = 128
        self.laser_range_max = 50  
        self.laser_angle_resolution = 0.05 * np.pi  
        self.laser_angle_half = 0.75 * np.pi  
        self.orientation = 0  
        self.position = np.zeros(2, dtype=np.int32)  
        self.move = [[0, 1],  
                     [1, 0],  
                     [0, -1],  
                     [-1, 0]] 
        self.explore_rate = 0  
        self.episode_steps = 0  

        self.ground_truth_map = np.load(abspath + '/map/{}.npy'.format(map_name))  
        self.ground_truth_map.flags.writeable = False  
        self.real_map = deepcopy(self.ground_truth_map)
        self.map = np.ones_like(self.real_map) * self.unknown_pixel
        self.grid_num = (self.real_map != self.unknown_pixel).sum()  
        self.global_map = np.zeros((self.explore_map_size, self.explore_map_size), dtype=np.uint8)
        self.local_map = np.zeros((self.explore_map_size, self.explore_map_size), dtype=np.uint8)

        self.random_obstacle = random_obstacle
        self.num_obstacles = 4
        if training:
            self.max_explore_rate = 0.99
        else:
            self.max_explore_rate = 1.0
        if render:
            plt.figure(figsize=(12, 12))
            plt.ion()
            self.Dim0 = []
            self.Dim1 = []
        self.action_dim = 3
        self.s_map_dim = (2, self.explore_map_size, self.explore_map_size)
        self.s_sensor_dim = (round(2 * self.laser_angle_half / self.laser_angle_resolution) + 2,)
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Dict({"s_map": spaces.Box(low=0, high=255, shape=self.s_map_dim, dtype=np.uint8), 
                                              "s_sensor": spaces.Box(low=0, high=1.0, shape=self.s_sensor_dim, dtype=np.float32)})  

        print("init {}".format(map_name))

    def update_map(self, ):
        """
            Use Ray-tracing algorithm to simulate the mapping process
            :return: laser
        """
        self.map[self.position[0], self.position[1]] = self.explored_pixel
        self.map[self.position[0], self.position[1]] = self.real_map[self.position[0], self.position[1]]
        laser = []
        for theta in np.arange(self.orientation * 0.5 * np.pi - self.laser_angle_half, self.orientation * 0.5 * np.pi + self.laser_angle_half + 1e-5, self.laser_angle_resolution):
            for r in range(1, self.laser_range_max + 1):
                dim0 = int(round(self.position[0] + r * np.sin(theta)))
                dim1 = int(round(self.position[1] + r * np.cos(theta)))

                self.map[dim0, dim1] = self.real_map[dim0, dim1]

                if self.real_map[dim0, dim1] == self.occupied_pixel:
                    break
            laser.append(np.sqrt((dim0 - self.position[0]) ** 2 + (dim1 - self.position[1]) ** 2))
        return np.array(laser, dtype=np.float32)

    def get_state(self):
        """
            Get current state
            :return: s, explore_rate
        """
        laser = self.update_map()  
        self.local_map = self.map[self.position[0] - self.local_map_size_half:self.position[0] + self.local_map_size_half, self.position[1] - self.local_map_size_half:self.position[1] + self.local_map_size_half]

        explore_map = (self.map != self.unknown_pixel) * self.explored_pixel
        explore_rate = explore_map.sum() / (self.grid_num * self.explored_pixel) 
        nonzero_index = np.nonzero(explore_map)
        dim0_min = nonzero_index[0].min()
        dim0_max = nonzero_index[0].max()
        dim1_min = nonzero_index[1].min()
        dim1_max = nonzero_index[1].max()
        global_map = explore_map[dim0_min:dim0_max + 1, dim1_min:dim1_max + 1]  
        global_map = cv2.resize(global_map, dsize=(self.explore_map_size, self.explore_map_size), interpolation=cv2.INTER_NEAREST) 

        position_0 = int((self.position[0] - dim0_min) * self.explore_map_size / (dim0_max - dim0_min))
        position_1 = int((self.position[1] - dim1_min) * self.explore_map_size / (dim1_max - dim1_min))
        global_map[np.clip(position_0 - 1, 0, self.explore_map_size):np.clip(position_0 + 2, 0, self.explore_map_size),
        np.clip(position_1 - 1, 0, self.explore_map_size):np.clip(position_1 + 2, 0, self.explore_map_size)] = self.agent_pixel

        self.global_map = global_map.astype(np.uint8)
        s_map = np.stack([self.global_map, self.local_map], axis=0)  
        s_sensor = np.concatenate([laser / self.laser_range_max, np.array([self.orientation / 4], dtype=np.float32)]) 

        s = {"s_map": s_map,
             "s_sensor": s_sensor}

        return s, explore_rate

    def get_info(self):
        """
            Get information
            :return: explore_rate, position, episode_steps
        """
        return {"explore_rate": self.explore_rate, "position": self.position, 'episode_steps': self.episode_steps}

    def random_init_obstacle(self):
        """
            Randomly initialize the position of the obstacle
            :return:
        """
        self.real_map = deepcopy(self.ground_truth_map)
        free_index = np.argwhere(self.real_map == self.free_pixel)
        for _ in range(self.num_obstacles):
            while True:
                obstacle_position = free_index[self.np_random.integers(len(free_index))]
                if (self.real_map[obstacle_position[0] - 1:obstacle_position[0] + 2, obstacle_position[1] - 1:obstacle_position[1] + 2]).sum() == self.free_pixel:
                    self.real_map[obstacle_position[0], obstacle_position[1]] = self.occupied_pixel
                    break

    def random_init_agent(self):
        """
            Randomly initialize the position and orientation of the agent
            :return:
        """
        self.orientation = self.np_random.integers(4)
        free_index = np.argwhere(self.real_map == self.free_pixel)
        agent_position = free_index[self.np_random.integers(len(free_index))]
        self.position[0] = agent_position[0]
        self.position[1] = agent_position[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        if self.random_obstacle:
            self.random_init_obstacle()
        self.random_init_agent()
        self.map = np.ones_like(self.real_map) * self.unknown_pixel
        self.select_target()
        s, explore_rate = self.get_state()
        self.explore_rate = explore_rate
        info = self.get_info()
        return s, info

    def select_target(self):
        target_x = np.random.randint(0, self.explore_map_size)
        target_y = np.random.randint(0, self.explore_map_size)
        self.target_path = self.plan_path(self.position, (target_x, target_y))
        return (target_x, target_y), self.target_path

    def plan_path(self, start, end):
        path = []
        current_position = list(start)
        while tuple(current_position) != end:
            if current_position[0] < end[0]:
                current_position[0] += 1
            elif current_position[0] > end[0]:
                current_position[0] -= 1
            if current_position[1] < end[1]:
                current_position[1] += 1
            elif current_position[1] > end[1]:
                current_position[1] -= 1
            path.append(tuple(current_position))
        return path
        
    def move(self, direction):
        if direction < 4: 
            new_position = (self.position[0] + self.move[direction][0], self.position[1] + self.move[direction][1])
            if self.is_within_bounds(new_position) and self.map[new_position] != self.occupied_pixel:
                self.position = new_position

    def is_within_bounds(self, position):
        return 0 <= position[0] < self.map_size and 0 <= position[1] < self.map_size

    def calculate_reward(self):
        explored = np.sum(self.map == self.explored_pixel)
        new_explore_rate = explored / np.size(self.map)
        reward = new_explore_rate - self.explore_rate
        self.explore_rate = new_explore_rate
        return reward

    def check_done(self):
        return self.explore_rate >= self.explored_rate_target or self.episode_steps >= self.episode_limit

    def step(self, action):
        self.episode_steps += 1
        if action == 0:  
            self.orientation = (self.orientation + 3) % 4
        elif action == 2:  
            self.orientation = (self.orientation + 1) % 4
        elif action == 1:  
            self.position += self.move[self.orientation]

        if self.real_map[self.position[0], self.position[1]] == self.occupied_pixel: 
            dead = True
            s = {"s_map": np.zeros(self.s_map_dim, dtype=np.uint8),
                 "s_sensor": np.zeros(self.s_sensor_dim, dtype=np.float32)}
            explore_rate = self.explore_rate
        else:
            dead = False
            s, explore_rate = self.get_state()
        if explore_rate > self.explore_rate: 
            r = np.clip((explore_rate ** 2 - self.explore_rate ** 2) * 10, 0, 1.0)
        else:
            r = -0.005

        if dead: 
            terminal = True
            r = -1.0
        elif explore_rate >= self.max_explore_rate: 
            terminal = True
            r += 1.0
        elif self.episode_steps == self.episode_limit:
            terminal = True
        else:
            terminal = False

        self.explore_rate = explore_rate  
        info = self.get_info()

        return s, r, terminal, False, info

    def render(self, mode='human'):
        plt.subplot(2, 2, 1)
        sns.heatmap(self.real_map, cmap='Greys', cbar=False)
        plt.scatter(self.position[1] + 0.5, self.position[0] + 0.5, c='r', marker='s', s=50)
        plt.subplot(2, 2, 2)
        sns.heatmap(self.map, cmap='Greys', cbar=False)
        plt.scatter(self.position[1] + 0.5, self.position[0] + 0.5, c='r', marker='s', s=50)
        plt.scatter(np.array(self.Dim1) + 0.5, np.array(self.Dim0) + 0.5, c='r', s=10)
        plt.subplot(2, 2, 3)
        sns.heatmap(self.local_map, cmap='Greys', cbar=False)
        plt.subplot(2, 2, 4)
        sns.heatmap(self.global_map, cmap='Greys', cbar=False)
        plt.show()
        plt.pause(0.05)
