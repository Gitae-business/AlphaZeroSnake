
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# class AlphaZeroNet(nn.Module):
#     def __init__(self, input_shape, num_actions, num_res_blocks=19):
#         super(AlphaZeroNet, self).__init__()
#         self.in_channels = input_shape[0]
        
#         self.conv1 = nn.Conv2d(self.in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(256)

#         self.res_blocks = self._make_layer(ResidualBlock, 256, num_res_blocks, stride=1)

#         # Policy head
#         self.policy_conv = nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=False)
#         self.policy_bn = nn.BatchNorm2d(2)
#         self.policy_fc = nn.Linear(2 * input_shape[1] * input_shape[2], num_actions)

#         # Value head
#         self.value_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
#         self.value_bn = nn.BatchNorm2d(1)
#         self.value_fc1 = nn.Linear(1 * input_shape[1] * input_shape[2], 256)
#         self.value_fc2 = nn.Linear(256, 1)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(256, out_channels, stride))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.res_blocks(out)

#         # Policy head
#         policy = F.relu(self.policy_bn(self.policy_conv(out)))
#         policy = policy.view(policy.size(0), -1)
#         policy = F.log_softmax(self.policy_fc(policy), dim=1)

#         # Value head
#         value = F.relu(self.value_bn(self.value_conv(out)))
#         value = value.view(value.size(0), -1)
#         value = F.relu(self.value_fc1(value))
#         value = torch.tanh(self.value_fc2(value))

#         return policy, value

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # GAP
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class PreActResidualBlock(nn.Module):
    """Pre-activation Residual block with optional SE block."""
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super(PreActResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)

        if self.use_se:
            out = self.se(out)

        return out + shortcut


class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape, num_actions, num_res_blocks=19, use_se=True):
        """
        Args:
            input_shape: (C, H, W)
            num_actions: number of possible actions
            num_res_blocks: number of residual blocks
            use_se: whether to use SE block
        """
        super(AlphaZeroNet, self).__init__()
        self.in_channels = input_shape[0]
        channels = 256

        self.conv1 = nn.Conv2d(self.in_channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.res_blocks = self._make_layer(
            PreActResidualBlock, channels, num_res_blocks, stride=1, use_se=use_se
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * input_shape[1] * input_shape[2], num_actions)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * input_shape[1] * input_shape[2], 256)
        self.value_fc2 = nn.Linear(256, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(out_channels, out_channels, s, use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.res_blocks(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value