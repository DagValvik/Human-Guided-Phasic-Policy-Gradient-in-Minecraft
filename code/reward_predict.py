import torch
import torch.nn as nn
import torch.optim as optim
from segment import Segment
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
        input_shape = (3, 256, 256)
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
        self.writer = SummaryWriter()

    @torch.no_grad()
    def reward(self, obs):
        obs = obs.to(self.device)
        r = self.core(obs)
        return r.item()

    def train_step(self, s1: Segment, s2: Segment, pref: list):
        # Convert the frames to tensors
        s1_frames = torch.stack(
            [
                torch.from_numpy(frame).permute(0, 3, 1, 2).squeeze()
                for frame in s1.frames
            ]
        ).to(self.device)
        s2_frames = torch.stack(
            [
                torch.from_numpy(frame).permute(0, 3, 1, 2).squeeze()
                for frame in s2.frames
            ]
        ).to(self.device)

        # Calculate the reward for each segment
        r1 = self.core(s1_frames)
        r2 = self.core(s2_frames)

        # Calculate the sum of rewards for each segment
        r1sum = torch.sum(r1)
        r2sum = torch.sum(r2)

        # Calculate the probability that segment 1 is preferred over segment 2
        # by comparing the sum of rewards
        p1 = torch.exp(r1sum) / (torch.exp(r1sum) + torch.exp(r2sum))
        p2 = torch.exp(r2sum) / (torch.exp(r1sum) + torch.exp(r2sum))

        # Calculate the loss
        pref_t = torch.tensor(
            pref, dtype=torch.float32, device=self.device
        )  # assuming pref is a list of floats
        loss = -(pref_t[0] * torch.log(p1) + pref_t[1] * torch.log(p2))

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, n_epochs):
        # Train for the specified number of epochs
        for epoch in tqdm(range(n_epochs), desc="Reward Predictor Training"):
            epoch_loss = 0
            for i, (s1, s2, pref) in enumerate(self.pref_queue):
                loss = self.train_step(s1, s2, pref)
                epoch_loss += loss

            # Print the average loss for the epoch
            print(f"Epoch {epoch}: {epoch_loss / len(self.pref_queue)}")

            # Log the average loss for the epoch
            self.writer.add_scalar(
                "Reward Predictor Loss",
                epoch_loss / len(self.pref_queue),
                epoch,
            )

    def save(self, path):
        torch.save(self.core.state_dict(), path)

    def load(self, path):
        self.core.load_state_dict(torch.load(path))
