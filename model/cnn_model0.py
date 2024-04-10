import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_size=(3, 32, 32), conv_kernel_sizes=[(3, 3), (3, 3)], conv_out_channels=[32, 64], fc_layer_sizes=[512], num_classes=10, dropout_rates=[0.5, 0.5]):
        super(CNNModel, self).__init__()
        # 修改卷积层以接受3通道输入
        self.conv1 = nn.Conv2d(input_size[0], conv_out_channels[0], kernel_size=conv_kernel_sizes[0])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv_out_channels[0], conv_out_channels[1], kernel_size=conv_kernel_sizes[1])

        self.fc_input_dim = self._compute_fc_input_dim(input_size, conv_kernel_sizes)
        # 修改全连接层的维度和数量
        self.fc1 = nn.Linear(self.fc_input_dim, fc_layer_sizes[0])
        self.dropout = nn.Dropout(dropout_rates[0])
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

    def _compute_fc_input_dim(self, input_size, conv_kernel_sizes):
        with torch.no_grad():
            x = torch.ones((1, *input_size))  # 创建一个虚拟的输入张量
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            fc_input_dim = x.size(1) * x.size(2) * x.size(3)  # 计算全连接层输入数据的维度
            return fc_input_dim

