import numpy as np
import matplotlib.pyplot as plt

def load_and_process_data(file_path):
    # 加载数据
    try:
        explore_rates = np.load(file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # 检查数据类型并尝试转换为浮点数
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
    # 计算基本统计数据
    mean_rate = np.mean(explore_rates)
    std_rate = np.std(explore_rates)
    min_rate = np.min(explore_rates)
    max_rate = np.max(explore_rates)
    
    print(f"Average exploration rate: {mean_rate}")
    print(f"Standard deviation of exploration rate: {std_rate}")
    print(f"Minimum exploration rate: {min_rate}")
    print(f"Maximum exploration rate: {max_rate}")

    # 数据可视化
    plt.figure(figsize=(10, 5))
    plt.plot(explore_rates, label='Exploration Rate')
    plt.title('Exploration Rate Progression')
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

# 指定文件路径
file_path = 'data_train/CCPPO_env_v1_train_map_l5_number_1_seed_1_rate.npy'

# 调用函数
explore_rates = load_and_process_data(file_path)
if explore_rates is not None:
    analyze_data(explore_rates)
