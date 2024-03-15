import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.cnn_model import CNNModel
import numpy as np
import torch.nn as nn

# from train import print_and_save
from utils.hyperparameters import CONV_KERNEL_SIZES, DROPOUT_RATES, LEARNING_RATE_STEPS

best_accuracy_global = 0
best_hyperparams_global = {}

file_path = "dqn_cnn_tuning_log2.txt"

def print_and_save(text, file_path):
    print(text)
    with open(file_path, "a") as file:
        file.write(text + "\n")

class CNNTuningEnvironment:
    def __init__(self):
        # 定义超参数空间
        self.conv_kernel_sizes = [(1, 1), (3, 3), (5, 5)]
        self.fc_layer_sizes = [128, 256, 512, 1024]
        self.dropout_rates = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.batch_sizes = [16, 32, 64, 128]
        self.learning_rates = np.linspace(0.0001, 0.01, num=10)  # 可以根据需要调整num的值
        self.momentums = np.linspace(0.5, 1, num=5)  # 同上

        # 添加缺失的属性定义
        self.num_kernel_options = len(self.conv_kernel_sizes)
        self.num_fc_layer_options = len(self.fc_layer_sizes)
        self.num_dropout_options = len(self.dropout_rates)
        self.num_batch_size_options = len(self.batch_sizes)
        self.num_lr_options = len(self.learning_rates)
        self.num_momentum_options = len(self.momentums)

        # 数据加载
        self.train_loader = DataLoader(datasets.MNIST('data', train=True, download=True,
                                                       transform=transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))
                                                       ])),
                                       batch_size=64, shuffle=True)
        self.test_loader = DataLoader(datasets.MNIST('data', train=False,
                                                     transform=transforms.Compose([
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.1307,), (0.3081,))
                                                     ])),
                                      batch_size=1000, shuffle=True)
        # 初始化上一次的性能
        self.last_performance = None  # 上一次的性能初始化为None
        self.last_hyperparams = None
        # self.max_iterations = 50  # 最大迭代次
        # self.iteration_count = 0  # 初始化迭代计数为0
        self.best_accuracy = 0  # 初始化最佳准确率
        self.best_hyperparams = {}  # 初始化最佳超参数

    def step(self, action):
        # 计算超参数索引
        num_kernel_options = len(self.conv_kernel_sizes)
        num_fc_layer_options = len(self.fc_layer_sizes)
        num_dropout_options = len(self.dropout_rates)
        num_batch_size_options = len(self.batch_sizes)
        num_lr_options = len(self.learning_rates)
        num_momentum_options = len(self.momentums)

        total_actions = num_kernel_options * num_fc_layer_options * num_dropout_options * num_batch_size_options * num_lr_options * num_momentum_options

        # print(f"step方法中：action: {action}, total_actions: {total_actions}")
        # 确保动作在有效范围内
        action = max(0, min(action, total_actions - 1))

        if action < 0 or action >= total_actions:
            raise ValueError("Action out of bounds")

        # 解码动作
        kernel_idx = action // (self.num_fc_layer_options * self.num_dropout_options * self.num_batch_size_options * self.num_lr_options * self.num_momentum_options)
        fc_layer_idx = (action // (self.num_dropout_options * self.num_batch_size_options * self.num_lr_options * self.num_momentum_options)) % self.num_fc_layer_options
        dropout_idx = (action // (self.num_batch_size_options * self.num_lr_options * self.num_momentum_options)) % self.num_dropout_options
        batch_size_idx = (action // (self.num_lr_options * self.num_momentum_options)) % self.num_batch_size_options
        lr_idx = (action // self.num_momentum_options) % self.num_lr_options
        momentum_idx = action % self.num_momentum_options

        # 解码超参数值
        conv_kernel_size = self.conv_kernel_sizes[kernel_idx]
        fc_layer_size = self.fc_layer_sizes[fc_layer_idx]
        dropout_rate = self.dropout_rates[dropout_idx]
        batch_size = self.batch_sizes[batch_size_idx]
        learning_rate = self.learning_rates[lr_idx]
        momentum_rate = self.momentums[momentum_idx]
        # print(f"step方法中：conv_kernel_size: {conv_kernel_size}, fc_layer_size: {fc_layer_size}, dropout_rate: {dropout_rate}, batch_size: {batch_size}, learning_rate: {learning_rate}, momentum_rate: {momentum_rate}")

        # 使用解码的超参数值配置和训练模型
        current_performance = self.train_and_evaluate_model(conv_kernel_size, fc_layer_size, dropout_rate, batch_size, learning_rate, momentum_rate)
        # current_performance = current_performance_tuple[0]
        # best_accuracy_global = current_performance_tuple[1]
        # best_hyperparams_global = current_performance_tuple[2]

        # 如果是第一次运行，没有之前的性能可比较，可以选择给予中性或稍微正向的初始奖励
        if self.last_performance is None:
            reward = 0  # 或者根据你的策略设定一个初始值
        else:
            reward = current_performance - self.last_performance  # 当前性能与上一次性能的差值作为奖励


        # 更新迭代次数
        # self.iteration_count += 1

        # print_and_save(f"iteration_count: {self.iteration_count}", file_path)
        # 判断是否达到终止条件，如果当前性能低于上一次的0.99倍，或者迭代次数达到最大值，则结束
        # or self.iteration_count >= self.max_iterations
        if self.last_performance is not None and (current_performance < self.last_performance * 0.98):
            # print("self.iteration_count: ", self.iteration_count)
            done = True
        else:
            done = False

        # 更新上一次的性能记录
        self.last_performance = current_performance
        self.last_hyperparams = {'conv_kernel_size': conv_kernel_size, 'fc_layer_size': fc_layer_size, 'dropout_rate': dropout_rate,
                                'batch_size': batch_size, 'learning_rate': learning_rate, 'momentum_rate': momentum_rate}
        # 返回新的状态（在本例中，状态可能不直接依赖于动作），奖励和是否结束
        return np.array([kernel_idx, fc_layer_idx, dropout_idx, batch_size_idx, lr_idx, momentum_idx]), reward, done

    # 使用act()方法选择动作后，调用step()方法执行动作(修改超参数)并获取下一个状态和奖励，这里是训练模型并获取性能
    def train_and_evaluate_model(self, conv_kernel_size, fc_layer_size, dropout_rate, batch_size, learning_rate, momentum_rate):
        # print(
        #     f"Training with params - Conv kernel size: {conv_kernel_size}, FC layer size: {fc_layer_size}, Dropout rate: {dropout_rate}, Batch size: {batch_size}, Learning rate: {learning_rate}, Momentum: {momentum_rate}")

        global best_accuracy_global
        global best_hyperparams_global
        # 数据准备
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 或保留为一个大批量，具体取决于你的评估需要

        # 模型定义
        model = CNNModel(conv_kernel_sizes=[conv_kernel_size, conv_kernel_size], fc_layer_sizes=[fc_layer_size],
                         dropout_rates=[dropout_rate])
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_rate)

        # 训练过程
        model.train()
        # 这个循环表示训练过程将遍历整个数据集5次。
        for epoch in range(15):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), target.to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 评估过程
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), target.to(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # print(f"Using device: {device}")
                output = model(data)
                # print(f"Model output shape: {output.size()}, Target shape: {target.size()}")
                pred = output.argmax(dim=1)  # 移除keepdim=True使pred变为一维
                correct += pred.eq(target.view(-1)).sum().item()

        accuracy = correct / len(test_loader.dataset)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_hyperparams = {'conv_kernel_size': conv_kernel_size, 'fc_layer_size': fc_layer_size, 'dropout_rate': dropout_rate,
                                    'batch_size': batch_size, 'learning_rate': learning_rate, 'momentum_rate': momentum_rate}

        return accuracy

    def reset(self):
        # 这里假设返回的是随机选择的初始状态
        # 确保根据你的状态空间调整
        state = np.array([np.random.randint(self.num_kernel_options),
                          np.random.randint(self.num_fc_layer_options),
                          np.random.randint(self.num_dropout_options),
                          np.random.randint(self.num_batch_size_options),
                          np.random.randint(self.num_lr_options),
                          np.random.randint(self.num_momentum_options)])
        self.last_performance = None  # 重置上一次的性能
        return state
