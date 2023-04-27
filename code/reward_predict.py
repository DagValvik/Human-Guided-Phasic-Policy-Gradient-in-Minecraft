import logging
import queue

import torch
import torch.nn as nn
import torch.optim as optim
from segment import Segment


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
            nn.Conv2d(3, 16, kernel_size=7, stride=3),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16) if batchnorm else nn.Identity(),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def create_classifier(self):
        input_shape = (3, 128, 128)
        output_size = self.calculate_output_size(input_shape)
        layers = [
            nn.Linear(output_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1)
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

    @torch.no_grad()
    def reward(self, obs):
        obs = obs.to(self.device)
        r = self.core(obs)
        return r.item()

    def train_step(self, s1: Segment, s2, pref: list):
        # Sum the rewards for each segment
        r1sum = sum(s1.rewards)
        r2sum = sum(s2.rewards)

        # Calculate the probability that segment 1 is preferred over segment 2
        # by comparing the sum of rewards
        p1 = torch.exp(torch.tensor(r1sum, requires_grad=True)) / (
            torch.exp(torch.tensor(r1sum, requires_grad=True))
            + torch.exp(torch.tensor(r2sum, requires_grad=True))
        )
        p2 = torch.exp(torch.tensor(r2sum, requires_grad=True)) / (
            torch.exp(torch.tensor(r1sum, requires_grad=True))
            + torch.exp(torch.tensor(r2sum, requires_grad=True))
        )

        # Calculate the loss
        loss = -(
            torch.tensor(pref[0], requires_grad=True) * torch.log(p1)
            + torch.tensor(pref[1], requires_grad=True) * torch.log(p2)
        )

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, max_iterations=None):
        iteration = 0
        while True:
            try:
                # Get a segment pair from the preference queue if there is any
                s1, s2, pref = self.pref_queue.get(timeout=1)
                loss = self.train_step(s1, s2, pref)
                logging.debug("Trained on preference %s", pref)
                logging.debug("Loss: %f", loss)

                # Increment the iteration counter and break if the maximum number of iterations is reached
                if max_iterations is not None:
                    iteration += 1
                    if iteration >= max_iterations:
                        break
            except queue.Empty:
                # If the preference queue is empty, continue waiting for preferences
                continue

    def save(self, path):
        torch.save(self.core.state_dict(), path)

    def load(self, path):
        self.core.load_state_dict(torch.load(path))
