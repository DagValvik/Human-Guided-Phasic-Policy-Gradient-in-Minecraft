from queue import Queue
from random import shuffle

import numpy as np


class PreferenceInterface:
    """
    A interface for showing the user segment pairs (videos) taken randomly from the segment queue and asking them
    to choose the preferred one. The user's choice is put in a preference queue.
    """

    def __init__(self, segment_queue: Queue, pref_queue: Queue):
        """
        :param segment_queue: The segment queue to get segments from
        :param pref_queue: The preference queue to put the user's preferences in
        """
        self.segment_queue = segment_queue
        self.pref_queue = pref_queue

    def shuffle_segment_queue(self):
        """
        Shuffle the segment queue to get random segments. Queue does not support shuffling, so we have to
        get all the segments, shuffle them, and put them back in the queue. This is not thread-safe, so we use locks.
        """
        raise NotImplementedError
        
    def run(self):
        """
        Run the preference interface.
        """
        while True:
            # Get a segment from the segment queue
            s1 = self.segment_queue.get(block=True)
            # Get anothersegment from the segment queue
            s2 = self.segment_queue.get(block=True)
            # Show the user the segment pair
            pref = self.show_segment_pair(s1, s2)
            # Put the user's preference in the preference queue
            self.pref_queue.put((s1, s2, pref))

    def show_segment_pair(self, s1, s2):
        """
        Show the user a segment pair and ask them to choose the preferred one.
        :param s1: The first segment
        :param s2: The second segment
        :return: The user's preference
        """
        raise NotImplementedError


class Segment:
    """
    A video recording of an agent's behavior in the environment,
    consisting of a sequence of frames and the rewards it received during those frames.
    """

    def __init__(self):
        """
        :param frames: The frames of the segment
        :param rewards: The rewards of the segment
        """
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
            self.hash = hash(np.array(self.frames).tostring())

    def __len__(self):
        """
        :return: The length of the segment
        """
        return len(self.frames)
