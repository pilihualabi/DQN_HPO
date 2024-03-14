import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import torch.nn.functional as F

# from train import print_and_save

file_path = "dqn_cnn_tuning_log2.txt"

def print_and_save(text, file_path):
    print(text)
    with open(file_path, "a") as file:
        file.write(text + "\n")

# class DQNNetwork(nn.Module):
#     def __init__(self, input_size, output_size=150, conv_kernel_sizes=[(3, 3)], fc_layer_sizes=[128, 64],
#                  dropout_rate=0.5):
#         super(DQNNetwork, self).__init__()
#
#         # 修改输入张量的尺寸
#         print(f"输入尺寸: {input_size}")
#
#         # 卷积层定义
#         self.conv_layers = nn.ModuleList()
#         in_channels = input_size[0]  # 假设input_size包含通道数
#         for idx, kernel_size in enumerate(conv_kernel_sizes):
#             out_channels = 32 * (2 ** idx)
#             self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
#             in_channels = out_channels
#         print(f"1self.conv_layers: {self.conv_layers}")
#
#         # 全连接层定义
#         self.fc_layers = nn.ModuleList()
#         num_features = self._get_conv_output(input_size)
#         for out_features in fc_layer_sizes:
#             self.fc_layers.append(nn.Linear(num_features, out_features))
#             num_features = out_features
#
#         # Dropout层
#         self.dropout = nn.Dropout(dropout_rate)
#
#         # 输出层
#         self.out = nn.Linear(fc_layer_sizes[-1], output_size)
#
#     def forward(self, x):
#         print(f"forward输入尺寸: {x.size()}")
#         # 应用卷积层和激活函数
#         for conv_layer in self.conv_layers:
#             x = F.relu(conv_layer(x))
#             print(f"卷积层输出尺寸: {x.size()}")
#             # 检查是否适合池化
#             if x.dim() > 3 and x.size(2) > 1 and x.size(3) > 1:  # 确保x具有至少4个维度
#                 x = F.max_pool2d(x, kernel_size=2)  # 示例使用2x2的最大池化
#                 print(f"池化层输出尺寸: {x.size()}")
#             else:
#                 print("跳过池化层，因为尺寸太小或维度不足")
#
#         print(f"卷积层和池化层输出尺寸: {x.size()}")
#         # 准备进入全连接层
#         x = torch.flatten(x, 1)  # 展平除批次维度外的所有维度
#
#         print("self.fc_layers: ", self.fc_layers)
#         # 应用全连接层和Dropoutssh -p 32756 root@region-41.seetacloud.com
#         for fc_layer in self.fc_layers:
#             # 线性层
#             x = F.relu(fc_layer(x))
#             x = self.dropout(x)
#
#         # 应用输出层
#         x = self.out(x)
#         print_and_save(f"Model output shape before squeeze: {x.size()}", file_path)
#         return x
#
#     def _get_conv_output(self, shape):
#         bs = 1
#         input = torch.autograd.Variable(torch.rand(bs, *shape))
#         output_feat = self._forward_features(input)
#         n_size = output_feat.data.view(bs, -1).size(1)
#         print(f"get_conv_output全连接层输入尺寸: {n_size}")
#         return n_size
#
#     def _forward_features(self, x):
#         for conv_layer in self.conv_layers:
#             x = F.relu(conv_layer(x))
#             x = F.max_pool2d(x, kernel_size=2)
#             print(f"forward_features卷积层输出尺寸: {x.size()}")
#         return x

class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()

        # 嵌入层定义
        self.embedding_conv_kernel = nn.Embedding(num_embeddings=3, embedding_dim=4)
        self.embedding_fc_layer = nn.Embedding(num_embeddings=4, embedding_dim=4)
        self.embedding_dropout = nn.Embedding(num_embeddings=5, embedding_dim=4)
        self.embedding_batch_size = nn.Embedding(num_embeddings=4, embedding_dim=4)
        self.embedding_learning_rate = nn.Embedding(num_embeddings=10, embedding_dim=4)
        self.embedding_momentum = nn.Embedding(num_embeddings=5, embedding_dim=4)

        # 计算所有嵌入的总维度
        total_embedding_dim = 4 * 6  # 6 hyperparameters each with an embedding size of 4

        # 全连接层定义
        self.fc1 = nn.Linear(total_embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 12000)  # 假设我们想要一个Q值的单个输出值

    def forward(self, x):
        # 嵌入层，将每个输入的索引转换为嵌入向量
        embeddings = []
        embeddings.append(self.embedding_conv_kernel(x[:, 0]))
        embeddings.append(self.embedding_fc_layer(x[:, 1]))
        embeddings.append(self.embedding_dropout(x[:, 2]))
        embeddings.append(self.embedding_batch_size(x[:, 3]))
        embeddings.append(self.embedding_learning_rate(x[:, 4]))
        embeddings.append(self.embedding_momentum(x[:, 5]))

        # 连接所有嵌入
        x = torch.cat(embeddings, dim=1)  # 沿着第二个维度连接所有嵌入

        # 应用全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(f"模型的输出：{x}")
        # print(f"模型的输出形状: {x.size()}")

        return x


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                     epsilon_decay=0.995, conv_kernel_sizes=[(3, 3)], fc_layer_sizes=[128, 64], dropout_rate=0.5, batch_size=64, momentum=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.momentum = momentum
        # self.train_steps = 0  # 初始化训练次数为0
        # self.model = DQNNetwork(input_size=(1, 28, 28),  # 假定的输入尺寸，需要根据实际情况调整
        #                         output_size=action_size,
        #                         conv_kernel_sizes=conv_kernel_sizes,
        #                         fc_layer_sizes=fc_layer_sizes,
        #                         dropout_rate=dropout_rate)
        self.model = DQNNetwork()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # print(f"DQNNetwork output size: {self.model.out.out_features}")
        self.best_accuracy = 0  # 初始化最佳准确率
        self.best_hyperparams = {}

    #remember方法用于存储经验
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #act方法用于选择动作
    def act(self, state):
        # self.train_steps += 1
        # 探索，随机选择动作
        if np.random.rand() <= self.epsilon:
            print_and_save(f"Random action range: 0 to {self.action_size - 1}", file_path)
            # print(f"Random action range: 0 to {self.action_size - 1}")
            return random.randrange(self.action_size)  # 确保这里不会选择超出范围的动作
        # 利用模型预测动作
        else:
            print(f"act中的state: {state}")
            if state.ndim == 1:
                # 如果是一维的，增加一个维度
                state = torch.tensor(state).long().unsqueeze(0)
            elif state.ndim == 2:
                # 如果已经是二维的，直接转换
                state = torch.tensor(state).long()
            act_values = self.model(state) # 获取模型输出
            print(f"act_values 的值: {act_values}")
            action = torch.argmax(act_values, axis=1).item()  # 获取最大值的索引
            # 返回的action应该是一个在0到action_size-1之间的整数
            print_and_save(f"Predicted action: {action}", file_path)
            # print(f"Predicted action: {action}")
            return action

    # replay方法用于训练模型
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # 从记忆中随机抽取一批经验
        minibatch = random.sample(self.memory, self.batch_size)
        # minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # print(f"state: {state}")
            # 计算目标值
            target = reward
            # 判断是否到达终止态，如果没有到达终止态，需要计算目标值，如果到达终止态，目标值就是reward
            if not done:
                # print(f"action1: {action}")
                # print(f"next_state : {next_state}")
                next_state = torch.tensor(next_state).long()
                # next_state = torch.FloatTensor(next_state).unsqueeze(0)
                # print(f"next_state : {next_state}")
                # 使用target=reward+γ×maxQ(s',a')来计算目标值
                target = (reward + self.gamma * torch.max(self.model(next_state))).item()
                # print(f"target: {target}")

            # 将state值转换为能够输入模型的张量
            if state.ndim == 1:
                # 如果是一维的，增加一个维度
                state = torch.tensor(state).long().unsqueeze(0)
            elif state.ndim == 2:
                # 如果已经是二维的，直接转换
                state = torch.tensor(state).long()

            # state = torch.FloatTensor(state).unsqueeze(0)
            # state = torch.tensor(state).long().unsqueeze(0)
            # print(f"state: {state}")

            # 计算当前状态的预测值，预测值的形状是
            target_f = self.model(state)

            # print(f"target_f: {target_f}")
            # print(f"target: {target}")

            # 创建一个目标Q值Tensor，其初始值为target_f的副本
            target_q_values = target_f.clone().detach()

            # 更新对应于实际采取的动作的Q值为计算出的target值
            target_q_values[0, action] = target

            # 计算损失，这次是target_f和新的目标Q值Tensor之间的差异
            loss = nn.MSELoss()(target_f, target_q_values)

            # 梯度清零
            self.optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新模型权重
            self.optimizer.step()
            # self.optimizer.zero_grad()
            # # print(f"state size: {state.size()}")
            # loss = nn.MSELoss()(target_f, self.model(state))
            # # print(f"loss size: {loss.size()}")
            # loss.backward()
            # # print(f"loss size after backward: {loss.size()}")
            # self.optimizer.step()
        self.scheduler.step()
        # print("判断是否需要衰减epsilon前的epsilon值:", self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # print_and_save("进行了epsilon衰减", file_path)
            # print("进行了epsilon衰减")
        # else:
        #     print("epsilon已经低于最小值，不再进行衰减")
        print_and_save(f"当前epsilon值:{self.epsilon}", file_path)


    def save(self, filepath):
        save_dir = os.path.dirname(filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'memory': self.memory,
            # 'train_steps': self.train_steps,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'best_accuracy': self.best_accuracy,  # 保存最佳准确率
            'best_hyperparams': self.best_hyperparams  # 保存最佳模型状态
        }

        print(f"保存的episode_count: {self.episode_count}")
        torch.save(state, filepath)
        print(f"模型及训练状态已保存到{filepath}")

    def load(self, filepath):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)

            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.memory = checkpoint['memory']
            # self.train_steps = checkpoint.get('train_steps', 0)
            self.epsilon = checkpoint.get('epsilon', 1.0)  # 加载epsilon
            self.episode_count = checkpoint.get('episode_count', 0)  # 加载episode_count
            self.best_accuracy = checkpoint.get('best_accuracy', 0)
            self.best_hyperparams = checkpoint.get('best_hyperparams', {})

            print(f"模型从episode {self.episode_count} 加载成功。")
            return self.episode_count, self.best_accuracy, self.best_hyperparams, self.epsilon
        else:
            print(f"文件 {filepath} 不存在.")
            return 0, 0, {}, 1.0

