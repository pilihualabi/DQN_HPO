import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, input_size=(1, 28, 28), conv_kernel_sizes=[(3, 3), (3, 3)], conv_out_channels=[32, 64], fc_layer_sizes=[128], num_classes=10, dropout_rates=[0.5, 0.5]):
        super(CNNModel, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=conv_kernel_sizes[0])
        # 最大池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=conv_kernel_sizes[1])
        #
        # print(f"Conv1: {self.conv1}")
        # print(f"Pool: {self.pool}")
        # print(f"Conv2: {self.conv2}")

        self.fc_input_dim = self._compute_fc_input_dim(input_size, conv_kernel_sizes)
        # print(f"Calculated fc_input_dim: {self.fc_input_dim}")
        # 第一个全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, fc_layer_sizes[0])
        # dropout层
        self.dropout = nn.Dropout(dropout_rates[0])
        # 第二个全连接层
        self.fc2 = nn.Linear(fc_layer_sizes[0], num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _compute_fc_input_dim(self, input_size, conv_kernel_sizess):
        with torch.no_grad():
            # print("Computing fc_input_dim")
            x = torch.ones((1, *input_size))  # 创建一个虚拟的输入张量
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            # print(f"After Conv1 and Pool shape: {x.size()}")
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            # print(f"After Conv2 and Pool shape: {x.size()}")
            fc_input_dim = x.size(1) * x.size(2) * x.size(3)  # 计算全连接层输入数据的维度
            return fc_input_dim

