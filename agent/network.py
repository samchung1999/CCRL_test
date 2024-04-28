import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def orthogonal_init(layer, gain=math.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer

class Actor_Critic(nn.Module):
    def __init__(self, config):
        super(Actor_Critic, self).__init__()
        # Initialize layers for processing the map and sensor inputs
        self.map_layer = nn.Sequential(
            orthogonal_init(nn.Conv2d(config.s_map_dim[0], 8, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            orthogonal_init(nn.Conv2d(8, 16, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            orthogonal_init(nn.Linear(16 * 5 * 5, config.hidden_dim)),
            nn.ReLU(),
        )

        self.sensor_layer = nn.Sequential(
            orthogonal_init(nn.Linear(config.s_sensor_dim[0], config.hidden_dim)),
            nn.ReLU(),
        )

        # Output layers for the actor and critic
        self.actor_out = orthogonal_init(nn.Linear(config.hidden_dim * 2, config.action_dim), gain=0.01)
        self.critic_out = orthogonal_init(nn.Linear(config.hidden_dim * 2, 1), gain=1.0)

        # New output layer for destination prediction
        self.destination_out = orthogonal_init(nn.Linear(config.hidden_dim * 2, 2), gain=1.0)  # Output 2D coordinates

    def forward(self, s_map, s_sensor):
        # 现在的forward方法处理所有输出
        combined_features = self.get_feature(s_map, s_sensor)
        action_logits = self.actor_out(combined_features)
        state_value = self.critic_out(combined_features).squeeze(-1)
        destination_prediction = self.destination_out(combined_features)
        return action_logits, state_value, destination_prediction

    def get_logit_and_value(self, s_map, s_sensor):
        # 这个方法返回动作概率（logits）和状态价值
        action_logits, state_value, _ = self.forward(s_map, s_sensor)
        return action_logits, state_value

    def actor(self, s_map, s_sensor):
        # 这个方法返回动作概率（logits）
        action_logits, _, _ = self.forward(s_map, s_sensor)
        return action_logits

    def critic(self, s_map, s_sensor):
        # 这个方法返回状态价值
        _, state_value, _ = self.forward(s_map, s_sensor)
        return state_value
        
    def get_feature(self, s_map, s_sensor):
        s_map = self.map_layer(s_map.float() / 255.0)  # Normalize and process map
        s_sensor = self.sensor_layer(s_sensor)
        return torch.cat([s_map, s_sensor], dim=-1)
