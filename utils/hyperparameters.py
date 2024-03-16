import numpy as np

# 定义卷积核大小的选项
CONV_KERNEL_SIZES = [(1, 1), (3, 3), (5, 5)]

# 定义Dropout比率的选项
DROPOUT_RATES = [0.3, 0.4, 0.5, 0.6, 0.7]

# 定义fc层节点数的选项
FC_NODES = [128, 256, 512, 1024]

# 定义批量大小的选项
BATCH_SIZES = [16, 32, 64, 128]

# 定义学习率的选项
LEARNING_RATES = np.linspace(0.0001, 0.01, num=10)  # 可以根据需要调整num的值

# 定义动量的选项
MOMENTUMS = np.linspace(0.5, 1, num=5)  # 同上



# 用于Agent的超参数
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.99  # 折扣率
LEARNING_RATE = 0.001  # Agent学习率
MEMORY_SIZE = 100000  # 经验回放的容量
BATCH_SIZE = 32  # 从记忆中抽取的批处理大小
