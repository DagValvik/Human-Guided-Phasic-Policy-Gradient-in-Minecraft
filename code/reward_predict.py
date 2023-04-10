import logging

import torch
import torch.nn as nn
import torch.optim as optim


class RewardPredictorCore(nn.Module):
    def __init__(self, batchnorm=True, dropout=0.5):
        super(RewardPredictorCore, self).__init__()
        self.feature_extractor = self.create_feature_extractor(
            batchnorm, dropout
        )
        self.classifier = self.create_classifier()

    def calculate_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.feature_extractor(dummy_input)
            return dummy_output.view(1, -1).size(1)

    def create_feature_extractor(self, batchnorm, dropout):
        layers = [
            nn.Conv2d(3, 32, kernel_size=7, stride=3),
            nn.BatchNorm2d(32) if batchnorm else nn.Identity(),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def create_classifier(self):
        input_shape = (3, 360, 640)
        output_size = self.calculate_output_size(input_shape)
        layers = [
            nn.Linear(output_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1)


class RewardPredictorNetwork(nn.Module):
    """
    Args:
        nn (_type_): _description_
    """

    def __init__(self, batchnorm=True, dropout=0.5, pref_queue=None):
        super(RewardPredictorNetwork, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.core = RewardPredictorCore(batchnorm, dropout).to(self.device)
        self.optimizer = optim.Adam(self.core.parameters(), lr=0.001)
        self.pref_queue = pref_queue

    def reward(self, obs):
        obs = obs.to(self.device)
        r = self.core(obs)
        return r

    def train_step(self, s1, s2, pref):
        # Each segment can be many frames, calculate the reward for each frame
        # and sum the rewards
        s1 = s1.to(self.device)  # Move s1 to the GPU
        s2 = s2.to(self.device)  # Move s2 to the GPU
        r1 = self.reward(s1)
        r2 = self.reward(s2)
        r2sum = r2.sum()
        r1sum = r1.sum()
        # Calculate the probability that segment 1 is preferred over segment 2
        # by comparing the sum of rewards
        p1 = torch.exp(r1sum) / (torch.exp(r1sum) + torch.exp(r2sum))
        p2 = torch.exp(r2sum) / (torch.exp(r1sum) + torch.exp(r2sum))
        # Calculate the loss
        loss = -(
            torch.tensor(pref[0]) * torch.log(p1)
            + torch.tensor(pref[1]) * torch.log(p2)
        )
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        # Get random segment pairs from the preference queue
        # and train the network on them
        while True:
            # Get a random segment pair from the preference queue if there is any
            s1, s2, pref = self.pref_queue.get()
            loss = self.train_step(s1, s2, pref)
            logging.debug("Trained on preference %s", pref)
            logging.debug("Loss: %f", loss)

    def save(self, path):
        torch.save(self.core.state_dict(), path)

    def load(self, path):
        self.core.load_state_dict(torch.load(path))
