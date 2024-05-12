import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import torch
import numpy as np
import argparse
from agent.network import Actor_Critic
import grid_simulator
import gym
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from utils.str2bool import str2bool
from torch.distributions import Categorical

abspath = os.path.dirname(os.path.abspath(__file__))

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.algorithm = config.algorithm
        self.env_version = config.env_version
        self.random_obstacle = config.random_obstacle
        self.train_map_name = config.train_map_name
        self.test_map_name = config.test_map_name
        self.number = config.number
        self.seed = config.seed
        self.model_index = config.model_index
        self.device = config.device

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        
        self.env = gym.make('ExploreEnv-{}'.format(self.env_version), map_name=self.test_map_name, random_obstacle=self.random_obstacle, training=False, render=False)
        self.env.reset(seed=self.seed)
        config.s_map_dim = self.env.observation_space["s_map"].shape
        config.s_sensor_dim = self.env.observation_space["s_sensor"].shape
        config.action_dim = self.env.action_space.n

        self.net = Actor_Critic(config).to(self.device)
        model_path = abspath + "/model/{}_env_{}_{}_number_{}_seed_{}_index_{}.pth".format(
            self.algorithm, self.env_version, self.train_map_name, self.number, self.seed, self.model_index)
        if self.model_index:
            print("load model...")
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("model index=0")
        print("model_path={}".format(model_path))
        print("test_map_name={}".format(self.test_map_name))

    def evaluate(self):
        evaluate_explore_rate = []
        steps_95 = []
        for evaluate_time in range(self.config.evaluate_times):
            s, info = self.env.reset()
            done = False
            win_95 = False
            while not done:
                a = self.choose_action(s)
                s_, r, done, _, info = self.env.step(a)
                s = s_
                if info['explore_rate'] >= 0.95 and not win_95:
                    steps_95.append(info['episode_steps'])
                    win_95 = True
            explore_rate = info['explore_rate']
            evaluate_explore_rate.append(explore_rate)
        return evaluate_explore_rate, steps_95

    def choose_action(self, s):
        with torch.no_grad():
            s_map = torch.from_numpy(s['s_map']).unsqueeze(0).to(self.device)
            s_sensor = torch.from_numpy(s['s_sensor']).unsqueeze(0).to(self.device)
            logit = self.net.actor(s_map, s_sensor)
            a = Categorical(logits=logit).sample()
            return a.cpu().item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--algorithm", type=str, default='CCPPO', help="The name of the algorithm")
    parser.add_argument("--env_version", type=str, default="v1", help="env_version")
    parser.add_argument("--train_map_name", type=str, default="all_maps", help="train_map_name")
    parser.add_argument("--test_map_name", type=str, default="test_map_l5", help="test_map_name")
    parser.add_argument("--random_obstacle", type=str2bool, nargs='?', const=True, default=True, help="Random obstacle presence")
    parser.add_argument("--number", type=int, default=1, help="number")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--evaluate_times", type=int, default=20, help="Evaluate times")
    parser.add_argument("--hidden_dim", type=int, default=32, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")


    config = parser.parse_args()

    all_results = {}
    for model_index in range(1, 301):
        config.model_index = model_index
        evaluator = Evaluator(config)
        rates, _ = evaluator.evaluate()
        mean_rate = np.mean(rates)
        all_results[model_index] = mean_rate
        print(f"Model Index: {model_index}, Average Exploration Rate: {mean_rate}")

    best_model_index = max(all_results, key=all_results.get)
    best_rate = all_results[best_model_index]
    print(f"Best Model Index: {best_model_index} with an Average Exploration Rate of {best_rate}")
