import random
from collections import deque

import cv2
import numpy as np


class PreferenceInterface:
    """
    A interface for showing the user segment pairs (videos) taken randomly from the segment queue and asking them
    to choose the preferred one. The user's choice is put in a preference queue.
    """

    def __init__(self, segment_queue: list, pref_queue: deque):
        """
        :param segment_queue: The segment queue to get segments from
        :param pref_queue: The preference queue to put the user's preferences in
        """
        self.segment_queue = segment_queue
        self.pref_queue = pref_queue
        self.task_name = "MineRLBasaltFindCave-v0"

    def shuffle_segment_queue(self):
        """
        Shuffle the segment queue to get random segments. Queue does not support shuffling, so we have to
        get all the segments, shuffle them, and put them back in the queue. This is not thread-safe, so we use locks.
        """
        raise NotImplementedError

    def add_segment(self, segment):
        """
        Add a segment to the segment queue.
        :param segment: The segment to add
        """
        self.segment_queue.append(segment)

    def add_segments(self, segments):
        """
        Add multiple segments to the segment queue.
        :param segments: The segments to add
        """
        self.segment_queue.extend(segments)

    def get_preferences(self):
        """
        Get the user's preferences.
        """
        while len(self.segment_queue) > 1:
            # Sample random segments from the list
            s1, s2 = random.sample(self.segment_queue, 2)
            # Show the user the segment pair
            pref = self.get_user_preference(s1, s2)
            # Put the user's preference in the preference queue
            self.pref_queue.append((s1, s2, pref))

    def show_segment_pair(self, s1, s2):
        """
        Show the user a segment pair in a loop until the user presses a key.
        :param s1: The first segment
        :param s2: The second segment
        :return: The user's preference
        """
        # Combine the two segments horizontally
        combined_frames = []
        for f1, f2 in zip(s1.frames, s2.frames):
            f1 = cv2.cvtColor(np.squeeze(f1), cv2.COLOR_RGB2BGR)
            f2 = cv2.cvtColor(np.squeeze(f2), cv2.COLOR_RGB2BGR)
            combined_frame = np.hstack((f1, f2))
            combined_frames.append(combined_frame)

        key = -1
        # Loop the segments until the user presses a key
        while key == -1:
            # Show the combined video
            for frame in combined_frames:
                cv2.imshow(self.task_name, frame)
                key = cv2.waitKey(1000 // 30)  # Assuming 30 FPS
                if key != -1:
                    break

        cv2.destroyAllWindows()

    def get_user_preference(self, s1, s2):
        """
        Get the user's preference between two segments.
        :param s1: The first segment
        :param s2: The second segment
        :return: The user's preference
        """
        self.show_segment_pair(s1, s2)

        # Print the instructions
        print("Enter your preference:")
        print("1 - First segment")
        print("2 - Second segment")
        print("3 - Equal preference")
        print("4 - Incomparable")

        # Get the user's preference
        while True:
            pref = input("Your choice: ").strip()

            if pref == "1":
                return [1.0, 0.0]  # first segment preferred
            elif pref == "2":
                return [0.0, 1.0]  # second segment preferred
            elif pref == "3":
                return [0.5, 0.5]  # equal preference
            elif pref == "4":
                return [-1.0, -1.0]  # incomparable preference
            else:
                print("Invalid input. Please enter 1, 2, 3, or 4.")
