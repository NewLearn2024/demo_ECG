import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()

        # Output: [32, 183]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        # Output: [32, 90]
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # Output: [32, 43]
        self.pool2 = nn.MaxPool1d(kernel_size=6, stride=2)

        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.leaky_relu3 = nn.LeakyReLU()
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # Output: [32, 20]
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.leaky_relu4 = nn.LeakyReLU()
        self.conv9 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # Output: [32, 8]
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=2)

        self.conv10 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.leaky_relu5 = nn.LeakyReLU()
        self.conv11 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # Output: [32, 2]
        self.pool5 = nn.MaxPool1d(kernel_size=6, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.leaky_relu1(self.conv2(x))
        x = x + self.conv3(x1)
        x = self.pool1(x)

        x = self.conv4(x)
        x1 = self.leaky_relu2(x)
        x = x + self.conv5(x1)
        x = self.pool2(x)

        x = self.conv6(x)
        x1 = self.leaky_relu3(x)
        x = x + self.conv7(x1)
        x = self.pool3(x)

        x = self.conv8(x)
        x1 = self.leaky_relu4(x)
        x = x + self.conv9(x1)
        x = self.pool4(x)

        x = self.conv10(x)
        x1 = self.leaky_relu5(x)
        x = x + self.conv11(x1)
        x = self.pool5(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x
