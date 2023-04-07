import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CustomCNN(nn.Module):
    def __init__(self, input_channels, batchnorm=True, dropout=0.5):
        super(CustomCNN, self).__init__()
        self.feature_extractor = self.create_feature_extractor(
            input_channels, batchnorm, dropout
        )
        self.classifier = self.create_classifier()

    def create_feature_extractor(self, input_channels, batchnorm, dropout):
        layers = [
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def create_classifier(self):
        layers = [
            nn.Linear(16 * 45 * 80, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.squeeze()


# # For single RGB image
# net_single = CustomCNN(input_channels=3)
# # For stack of 5 RGB images
# net_stacked = CustomCNN(input_channels=3*5)


class RewardPredictorNetwork(nn.Module):
    def __init__(self, input_channels, dropout, batchnorm, lr):
        super(RewardPredictorNetwork, self).__init__()
        self.core_network = CustomCNN(input_channels=input_channels, dropout=dropout, batchnorm=batchnorm)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.core_network.parameters(), lr=lr)

    def forward(self, s1, s2, pref, training):
        self.core_network.train(training)

        _r1 = self.core_network(s1)
        _r2 = self.core_network(s2)

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


# # For a single RGB image
# reward_predictor_single = RewardPredictorNetwork(input_channels=3, dropout=0.5, batchnorm=True, lr=0.001)

# # For a stack of 5 RGB images
# reward_predictor_stacked = RewardPredictorNetwork(input_channels=3*5, dropout=0.5, batchnorm=True, lr=0.001)
