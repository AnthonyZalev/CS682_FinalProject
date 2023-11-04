import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes, first_out_channel):
        super().__init__()
        
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=first_out_channel//2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(first_out_channel//2)
        self.conv1_2 = nn.Conv2d(in_channels=first_out_channel//2, out_channels=first_out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(first_out_channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=first_out_channel, out_channels=first_out_channel*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(first_out_channel*2)
        self.conv2_2 = nn.Conv2d(in_channels=first_out_channel*2, out_channels=first_out_channel*2, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(first_out_channel*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=first_out_channel*2, out_channels=first_out_channel*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(first_out_channel*4)
        self.conv3_2 = nn.Conv2d(in_channels=first_out_channel*4, out_channels=first_out_channel*4, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(first_out_channel*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=first_out_channel*4, out_channels=first_out_channel*8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(first_out_channel*8)
        self.conv4_2 = nn.Conv2d(in_channels=first_out_channel*8, out_channels=first_out_channel*8, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(first_out_channel*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
                
        self.conv5 = nn.Conv2d(in_channels=first_out_channel*8, out_channels=first_out_channel*8, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=first_out_channel*8, out_channels=first_out_channel*8, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(first_out_channel*8*8, first_out_channel*16)
        self.bn5 = nn.BatchNorm1d(first_out_channel*16)
        self.fc2 = nn.Linear(first_out_channel*16, first_out_channel*8)
        self.bn6 = nn.BatchNorm1d(first_out_channel*8)
        self.fc3 = nn.Linear(first_out_channel*8, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        x = self.pool3(x)

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        # x = self.conv4_2(x)
        # x = self.bn4_2(x)
        # x = self.relu(x)
        # x = self.pool4(x)

        # x = self.conv5(x)
        # x = self.relu(x)
        # x = self.conv5_2(x)
        # x = self.relu(x)
        # x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn6(x)
        x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.fc3(x)
        # x = self.softmax(x)

        return x