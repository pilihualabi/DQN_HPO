import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import torch.nn.functional as F

# from train import print_and_save

file_path = "dqn_cnn_tuning_log.txt"

def print_and_save(text, file_path):
    print(text)
    with open(file_path, "a") as file:
        file.write(text + "\n")

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size=150, conv_kernel_sizes=[(3, 3)], fc_layer_sizes=[128, 64],
                 dropout_rate=0.5):
        super(DQNNetwork, self).__init__()

        # 卷积层定义
        self.conv_layers = nn.ModuleList()
        in_channels = input_size[0]  # 假设input_size包含通道数
        for idx, kernel_size in enumerate(conv_kernel_sizes):
            out_channels = 32 * (2 ** idx)
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            in_channels = out_channels

        # 全连接层定义
        self.fc_layers = nn.ModuleList()
        num_features = self._get_conv_output(input_size)
        for out_features in fc_layer_sizes:
            self.fc_layers.append(nn.Linear(num_features, out_features))
            num_features = out_features

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层
        self.out = nn.Linear(fc_layer_sizes[-1], output_size)

    def forward(self, x):
        # 应用卷积层和激活函数
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool2d(x, kernel_size=2)  # 示例使用2x2的最大池化

        # 准备进入全连接层
        x = torch.flatten(x, 1)  # 展平除批次维度外的所有维度

        # 应用全连接层和Dropout
        for fc_layer in self.fc_layers:
            x = F.relu(fc_layer(x))
            x = self.dropout(x)

        # 应用输出层
        x = self.out(x)
        print_and_save(f"Model output shape before squeeze: {x.size()}", file_path)
        # print(f"Model output shape before squeeze: {x.size()}")
        return x

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool2d(x, kernel_size=2)
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
        self.model = DQNNetwork(input_size=(1, 28, 28),  # 假定的输入尺寸，需要根据实际情况调整
                                output_size=action_size,
                                conv_kernel_sizes=conv_kernel_sizes,
                                fc_layer_sizes=fc_layer_sizes,
                                dropout_rate=dropout_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # print(f"DQNNetwork output size: {self.model.out.out_features}")

    #remember方法用于存储经验
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #act方法用于选择动作
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print_and_save(f"Random action range: 0 to {self.action_size - 1}", file_path)
            # print(f"Random action range: 0 to {self.action_size - 1}")
            return random.randrange(self.action_size)  # 确保这里不会选择超出范围的动作
        else:
            # 如果不是随机选择动作，也打印预测的动作值
            state = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state)
            action = torch.argmax(act_values, axis=1).item()
            print_and_save(f"Predicted action: {action}", file_path)
            # print(f"Predicted action: {action}")
            return action

    #replay方法用于训练模型
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        # minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state))).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            # print(f"Action index: {action}")
            # print(f"target_f size: {target_f.size()}")

            if target_f.dim() == 3 and target_f.size(1) == 1:  # 形状为 [batch_size, 1, action_size]
                target_f = target_f.squeeze(1)  # 调整为 [batch_size, action_size]
            elif target_f.dim() != 2:
                print_and_save(f"Unexpected target_f shape: {target_f.size()}", file_path)
                # print(f"Unexpected target_f shape: {target_f.size()}")
                # 这里可以添加更多的处理逻辑

            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        print_and_save(f"判断是否需要衰减epsilon前的epsilon值: {self.epsilon}", file_path)
        # print("判断是否需要衰减epsilon前的epsilon值:", self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print_and_save("进行了epsilon衰减", file_path)
            # print("进行了epsilon衰减")
        else:
            print("epsilon已经低于最小值，不再进行衰减")
        print("当前epsilon值:", self.epsilon)
        #
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    # load和save方法用于加载和保存模型
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        # 确保目录存在
        save_dir = os.path.dirname(name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.model.state_dict(), name)

    # def load(self, filepath):
    #     """
    #     加载模型状态、训练轮次、最佳准确率、优化器状态以及训练历史记录。
    #     """
    #     if os.path.isfile(filepath):
    #         checkpoint = torch.load(filepath)
    #         self.model.load_state_dict(checkpoint['state_dict'])
    #
    #         # 加载训练历史记录
    #         training_history = checkpoint.get('training_history', None)
    #
    #         return checkpoint['epoch'], checkpoint['best_accuracy'], checkpoint['optimizer'], training_history
    #     else:
    #         print(f"No checkpoint found at '{filepath}'")
    #         return None, None, None, None
    #
    # def save(self, filepath, epoch, best_accuracy, optimizer, training_history):
    #     """
    #     保存模型状态、训练轮次、最佳准确率、优化器状态以及训练历史记录。
    #     training_history: 一个字典，包含了训练过程中的统计信息，如每个epoch的损失和准确率。
    #     """
    #     save_dir = os.path.dirname(filepath)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #
    #     state = {
    #         'epoch': epoch,
    #         'state_dict': self.model.state_dict(),
    #         'best_accuracy': best_accuracy,
    #         'optimizer': optimizer.state_dict(),
    #         'training_history': training_history,  # 添加训练历史记录
    #     }
    #
    #     torch.save(state, filepath)


