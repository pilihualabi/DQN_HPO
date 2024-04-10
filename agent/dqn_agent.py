import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import torch.nn.functional as F

# from train import print_and_save

file_path = "dqn_cifar1.txt"


def print_and_save(text, file_path):
    print(text)
    with open(file_path, "a") as file:
        file.write(text + "\n")


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
                 epsilon_decay=0.995, conv_kernel_sizes=[(3, 3)], fc_layer_sizes=[128, 64], dropout_rate=0.5,
                 batch_size=32, momentum=0.9):
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

        self.eval_network = DQNNetwork()
        self.target_network = DQNNetwork()
        self.target_network.load_state_dict(self.eval_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.SGD(self.eval_network.parameters(), lr=learning_rate, momentum=momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # print(f"DQNNetwork output size: {self.model.out.out_features}")
        self.best_accuracy = 0  # 初始化最佳准确率
        self.best_hyperparams = {}
        self.step_done = 0
        self.target_update = 10

    # remember方法用于存储经验
    def remember(self, state, action, reward, next_state, done):
        print(f"State shape: {np.array(state).shape}")  # 添加打印语句来检查状态的形状
        self.memory.append((state, action, reward, next_state, done))

    # act方法用于选择动作
    def act(self, state):
        # 探索，随机选择动作
        if np.random.rand() <= self.epsilon:
            print_and_save(f"Random action range: 0 to {self.action_size - 1}", file_path)
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
            act_values = self.eval_network(state)
            print(f"act_values 的值: {act_values}")
            action = torch.argmax(act_values, axis=1).item()  # 获取最大值的索引
            print_and_save(f"Predicted action: {action}", file_path)
            return action

    # replay方法用于训练模型
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # 从记忆中随机抽取一批经验
        minibatch = random.sample(self.memory, self.batch_size)
        #         # print(f"Sampled state shape: {np.array([t[0] for t in minibatch]).shape}")
        #         for state, action, reward, next_state, done in minibatch:
        #             print(f"States array shape before tensor conversion: {states.shape}")
        #             states = torch.tensor([t[0] for t in minibatch]).long()
        #             actions = torch.tensor([t[1] for t in minibatch]).long().unsqueeze(1)  # 添加额外的维度以便后续使用gather
        #             rewards = torch.tensor([t[2] for t in minibatch])
        #             next_states = torch.tensor([t[3] for t in minibatch]).long()
        #             dones = torch.tensor([t[4] for t in minibatch])

        #             if not done:
        #                 target = reward + self.gamma * torch.max(self.target_network(next_state).detach())
        #             else:
        #                 target = reward

        #             # 使用eval网络预测当前状态的Q值
        #             current_q_values = self.eval_network(states).gather(1, actions)

        #             # 使用target网络预测下一个状态的Q值
        #             next_q_values = self.target_network(next_states).max(1)[0].detach()
        #             targets = rewards + (self.gamma * next_q_values * (1 - dones))
        #             # 计算MSE损失
        #             loss = F.mse_loss(current_q_values, target)
        for state, action, reward, next_state, done in minibatch:
            # 判断是否到达终止态，如果没有到达终止态，需要计算目标值，如果到达终止态，目标值就是reward
            if not done:
                next_state = torch.tensor(next_state).long()
                # 这里使用目标网络来计算目标Q值
                target = reward + self.gamma * torch.max(self.target_network(next_state)).item()
            else:
                target = reward

            # 将state值转换为能够输入模型的张量
            if state.ndim == 1:
                # 如果是一维的，增加一个维度
                state = torch.tensor(state).long().unsqueeze(0)
            elif state.ndim == 2:
                # 如果已经是二维的，直接转换
                state = torch.tensor(state).long()

            # 获取当前状态的预测值
            current_q_values = self.eval_network(state)

            # 创建与current_q_values形状相同的target_q_values张量
            target_q_values = current_q_values.clone()
            # 更新执行的动作对应的Q值
            target_q_values[0, action] = target

            #             print(f"current_q_values shape: {current_q_values.shape}")
            #             print(f"target:{target}")
            # print(f"target shape: {target.shape}")

            # 计算损失
            loss = nn.MSELoss()(current_q_values, target_q_values)

            # 梯度清零
            self.optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新模型权重
            self.optimizer.step()

        self.scheduler.step()
        self.step_done += 1
        # print("判断是否需要衰减epsilon前的epsilon值:", self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # 根据一定步数更新目标网络
        if self.step_done % self.target_update == 0:
            self.target_network.load_state_dict(self.eval_network.state_dict())
        print_and_save(f"当前epsilon值:{self.epsilon}", file_path)

        return loss.item()

    def save(self, filepath):
        save_dir = os.path.dirname(filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        state = {
            'state_dict': self.eval_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'memory': self.memory,
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

            self.eval_network.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.memory = checkpoint['memory']
            self.epsilon = checkpoint.get('epsilon', 1.0)  # 加载epsilon
            self.episode_count = checkpoint.get('episode_count', 0)  # 加载episode_count
            self.best_accuracy = checkpoint.get('best_accuracy', 0)
            self.best_hyperparams = checkpoint.get('best_hyperparams', {})

            print(f"模型从episode {self.episode_count} 加载成功。")
            return self.episode_count, self.best_accuracy, self.best_hyperparams, self.epsilon
        else:
            print(f"文件 {filepath} 不存在.")
            return 0, 0, {}, 1.0

