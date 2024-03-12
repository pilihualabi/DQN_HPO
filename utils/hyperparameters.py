# 定义卷积核大小的选项
CONV_KERNEL_SIZES = [(1, 1), (3, 3), (5, 5)]

# 定义Dropout比率的选项
DROPOUT_RATES = [0.3, 0.4, 0.5, 0.6, 0.7]

# 定义学习率的范围
LEARNING_RATE_MIN = 0.0001
LEARNING_RATE_MAX = 0.01
LEARNING_RATE_STEPS = 10

# 用于Agent的超参数
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.99  # 折扣率
LEARNING_RATE = 0.001  # Agent学习率
MEMORY_SIZE = 100000  # 经验回放的容量
BATCH_SIZE = 32  # 从记忆中抽取的批处理大小
