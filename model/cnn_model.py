import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, input_size=(3, 32, 32), conv_kernel_sizes=[(3, 3), (3, 3)], conv_out_channels=[64, 128, 256],
                 fc_layer_sizes=[512], num_classes=10, dropout_rates=[0.5, 0.5]):
        super(CNNModel, self).__init__()
        # # 修改卷积层以接受3通道输入
        #
        # self.conv1 = nn.Conv2d(input_size[0], conv_out_channels[0], kernel_size=conv_kernel_sizes[0])
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(conv_out_channels[0], conv_out_channels[1], kernel_size=conv_kernel_sizes[1])
        #
        # self.fc_input_dim = self._compute_fc_input_dim(input_size, conv_kernel_sizes)
        # # 修改全连接层的维度和数量
        # self.fc1 = nn.Linear(self.fc_input_dim, fc_layer_sizes[0])
        # self.dropout = nn.Dropout(dropout_rates[0])
        # self.fc2 = nn.Linear(fc_layer_sizes[0], num_classes)

        #         self.conv1 = nn.Conv2d(input_size[0], conv_out_channels[0], kernel_size=conv_kernel_sizes[0], padding=1, stride=1)
        #         self.conv2 = nn.Conv2d(conv_out_channels[0], conv_out_channels[1], kernel_size=conv_kernel_sizes[0], padding=1, stride=1)
        #         self.conv3 = nn.Conv2d(conv_out_channels[1], conv_out_channels[2], kernel_size=conv_kernel_sizes[0], padding=1, stride=1)

        #         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #         self.fc_input_dim = self._compute_fc_input_dim(input_size)

        #         self.fc1 = nn.Linear(self.fc_input_dim,  fc_layer_sizes[0])
        #         self.fc2 = nn.Linear( fc_layer_sizes[0], num_classes)

        #         self.dropout = nn.Dropout(dropout_rates[0])

        #     def forward(self, x):
        #         # x = F.relu(self.conv1(x))
        #         # x = self.pool(x)
        #         # x = F.relu(self.conv2(x))
        #         # x = self.pool(x)
        #         # x = x.view(-1, self.fc_input_dim)
        #         # x = F.relu(self.fc1(x))
        #         # x = self.dropout(x)
        #         # x = self.fc2(x)
        #         # return x

        #         x = F.relu(self.conv1(x))
        #         x = self.pool(x)
        #         x = F.relu(self.conv2(x))
        #         x = self.pool(x)
        #         x = F.relu(self.conv3(x))
        #         x = self.pool(x)
        #         x = x.view(-1, self.fc_input_dim)
        #         x = F.relu(self.fc1(x))
        #         x = self.dropout(x)
        #         x = self.fc2(x)
        #         return x

        #     def _compute_fc_input_dim(self, input_size, conv_kernel_sizes):
        #         with torch.no_grad():
        #             # x = torch.ones((1, *input_size))  # 创建一个虚拟的输入张量
        #             # x = F.relu(self.conv1(x))
        #             # x = self.pool(x)
        #             # x = F.relu(self.conv2(x))
        #             # x = self.pool(x)
        #             # fc_input_dim = x.size(1) * x.size(2) * x.size(3)  # 计算全连接层输入数据的维度

        #             x = torch.ones((1, *input_size))
        #             x = F.relu(self.conv1(x))
        #             x = self.pool(x)
        #             x = F.relu(self.conv2(x))
        #             x = self.pool(x)
        #             x = F.relu(self.conv3(x))
        #             x = self.pool(x)
        #             fc_input_dim = x.size(1) * x.size(2) * x.size(3)
        #             return fc_input_dim

        self.conv1 = nn.Conv2d(input_size[0], conv_out_channels[0], kernel_size=conv_kernel_sizes[0], padding=1,
                               stride=1)
        self.conv2 = nn.Conv2d(conv_out_channels[0], conv_out_channels[0], kernel_size=conv_kernel_sizes[0], padding=1,
                               stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(conv_out_channels[0], conv_out_channels[1], kernel_size=conv_kernel_sizes[0], padding=1,
                               stride=1)
        self.conv4 = nn.Conv2d(conv_out_channels[1], conv_out_channels[1], kernel_size=conv_kernel_sizes[0], padding=1,
                               stride=1)
        self.conv5 = nn.Conv2d(conv_out_channels[1], conv_out_channels[2], kernel_size=conv_kernel_sizes[0], padding=1,
                               stride=1)

        self.fc_input_dim = self._compute_fc_input_dim(input_size)

        self.fc1 = nn.Linear(self.fc_input_dim, fc_layer_sizes[0])
        self.fc2 = nn.Linear(fc_layer_sizes[0], num_classes)

        self.dropout = nn.Dropout(dropout_rates[0])

    def forward(self, x):
        z1 = self.conv1(x)
        z1.requires_grad_()
        x1 = torch.relu(z1)
        z2 = self.conv2(x1)
        z2.requires_grad_()
        x2 = torch.relu(z2)
        x = self.maxpool(x2)
        z3 = self.conv3(x)
        z3.requires_grad_()
        x3 = torch.relu(z3)
        z4 = self.conv4(x3)
        z4.requires_grad_()
        x4 = torch.relu(z4)
        x = self.maxpool(x4)
        z5 = self.conv5(x)
        z5.requires_grad_()
        x5 = torch.relu(z5)
        x = self.maxpool(x5)
        x = x.view(x.shape[0], -1)
        z6 = self.fc1(x)
        z6.requires_grad_()
        x6 = torch.relu(z6)
        out = self.fc2(x6)
        l = torch.sum(out)
        return out

    def _compute_fc_input_dim(self, input_size):
        # 临时模拟一个输入数据通过卷积层和池化层的过程来计算fc层的输入维度
        # 使用一个大小为(batch_size=1, channels, height, width)的假输入
        temp_input = torch.randn(1, *input_size)
        # 通过第一个卷积层和激活层
        temp_output = F.relu(self.conv1(temp_input))
        # 通过第二个卷积层和激活层，注意此处没有再次应用池化
        temp_output = F.relu(self.conv2(temp_output))
        # 应用第一次最大池化
        temp_output = self.maxpool(temp_output)
        # 通过第三个卷积层和激活层
        temp_output = F.relu(self.conv3(temp_output))
        # 通过第四个卷积层和激活层，注意此处没有再次应用池化
        temp_output = F.relu(self.conv4(temp_output))
        # 应用第二次最大池化
        temp_output = self.maxpool(temp_output)
        # 通过第五个卷积层和激活层
        temp_output = F.relu(self.conv5(temp_output))
        # 应用第三次最大池化
        temp_output = self.maxpool(temp_output)
        # 计算全连接层输入的维度
        fc_input_dim = temp_output.view(-1).size(0)
        return fc_input_dim

