import torch


class Segment:
    """
    A video recording of an agent's behavior in the environment,
    consisting of a sequence of frames and the rewards it received during those frames.
    """

    def __init__(self):
        self.frames = []
        self.rewards = []
        self.hash = None

    def add_frame(self, frame, reward):
        """
        Add a frame and its reward to the segment.
        :param frame: The frame
        :param reward: The reward
        """
        self.frames.append(frame)
        self.rewards.append(reward)

    def finish(self, sequence_id=None):
        """
        Finish the segment.
        """
        if sequence_id is not None:
            self.sequence_id = sequence_id
        else:
            self.hash = hash(torch.cat(self.frames).numpy().tobytes())

    def __len__(self):
        """
        :return: The length of the segment
        """
        return len(self.frames)
