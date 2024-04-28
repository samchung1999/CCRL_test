import numpy as np
import matplotlib.pyplot as plt

file_reward_path = 'data_train/CCPPO_env_v1_train_map_l2_number_1_seed_1_reward.npy'

data_reward = np.load(file_reward_path)

mean_reward = np.mean(data_reward)
std_reward = np.std(data_reward)
print("Average Reward:", mean_reward)
print("Standard Deviation of Reward:", std_reward)



plt.figure(figsize=(10, 5))
plt.plot(data_reward, label='Reward over Time')
plt.title('Reward Progression')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()
