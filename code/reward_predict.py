import torch
import torch.nn as nn
import torch.optim as optim


class CoreNetwork(nn.Module):
    def __init__(self, dropout, batchnorm):
        super(CoreNetwork, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=3),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.Dropout(dropout),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
        )

        # Update the input size for the first fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(
                33600, 64
            ),  # You may need to adjust this value based on the output size of the last convolutional layer
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(64, 1)

    def forward(self, s, training):
        x = s / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x.squeeze()


class RewardPredictorNetwork(nn.Module):
    def __init__(self, core_network, dropout, batchnorm, lr):
        super(RewardPredictorNetwork, self).__init__()
        self.core_network = core_network(dropout, batchnorm)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.core_network.parameters(), lr=lr)

    def forward(self, s1, s2, pref, training):
        s1_unrolled = s1.view(-1, 84, 84, 4)
        s2_unrolled = s2.view(-1, 84, 84, 4)

        _r1 = self.core_network(s1_unrolled, training)
        _r2 = self.core_network(s2_unrolled, training)

        r1 = _r1.view(s1.size(0), s1.size(1))
        r2 = _r2.view(s2.size(0), s2.size(1))

        rs1 = torch.sum(r1, dim=1)
        rs2 = torch.sum(r2, dim=1)

        rs = torch.stack([rs1, rs2], dim=1)
        pred = nn.functional.softmax(rs, dim=1)

        preds_correct = torch.eq(
            torch.argmax(pref, dim=1), torch.argmax(pred, dim=1)
        )
        accuracy = torch.mean(preds_correct.float())

        loss = self.loss_fn(rs, torch.argmax(pref, dim=1))

        return r1, r2, rs1, rs2, pred, accuracy, loss

    def train_step(self, s1, s2, pref, training):
        self.optimizer.zero_grad()
        r1, r2, rs1, rs2, pred, accuracy, loss = self.forward(
            s1, s2, pref, training
        )
        loss.backward()
        self.optimizer.step()
        return r1, r2, rs1, rs2, pred, accuracy, loss
