import numpy as np
import matplotlib.pyplot as plt

def load_and_process_data(file_path):
   
    try:
        explore_rates = np.load(file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    
    if explore_rates.dtype.kind not in 'iu':
        try:
            explore_rates = explore_rates.astype(float)
            print("Data converted to float successfully.")
        except ValueError:
            print("Data conversion failed. Data is not numeric.")
            return None
    else:
        print("Data is already numeric.")
    
    return explore_rates

def analyze_data(explore_rates):
    mean_rate = np.mean(explore_rates)
    std_rate = np.std(explore_rates)
    min_rate = np.min(explore_rates)
    max_rate = np.max(explore_rates)
    
    print(f"Average exploration rate: {mean_rate}")
    print(f"Standard deviation of exploration rate: {std_rate}")
    print(f"Minimum exploration rate: {min_rate}")
    print(f"Maximum exploration rate: {max_rate}")

   
    plt.figure(figsize=(10, 5))
    plt.plot(explore_rates, label='Exploration Rate')
    plt.title('Exploration Rate Progression')
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate')
    plt.legend()
    plt.grid(True)
    plt.show()


map_number = input("Enter the train map number (e.g., 1, 2, 3, 4, 5): ")
seed = input("Enter the seed number (e.g., 1): ")

# 构建地图名称和文件路径
train_map = f"train_map_l{map_number}"
file_path = f'data_train/CCPPO_env_v1_{train_map}_number_1_seed_{seed}_rate.npy'

# 调用函数
explore_rates = load_and_process_data(file_path)
if explore_rates is not None:
    analyze_data(explore_rates)
