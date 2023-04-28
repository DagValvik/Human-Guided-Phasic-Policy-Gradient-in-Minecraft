from collections import namedtuple
from typing import List

from torch.utils.data import DataLoader, Dataset

Memory = namedtuple(
    "Memory",
    [
        "agent_obs",
        "action",
        "action_log_prob",
        "reward",
        "done",
        "value",
        "next_value",
    ],
)
AuxMemory = namedtuple("Memory", ["agent_obs", "rewards", "old_values"])


class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple([d[ind] for d in self.data])


def create_dataloader(data, batch_size):
    dataset = ExperienceDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


Episode = List[Memory]
