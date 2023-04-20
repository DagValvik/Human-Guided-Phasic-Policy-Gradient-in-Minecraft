from queue import Queue

import cv2
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
        # Combine the two segments horizontally
        combined_frames = []
        for f1, f2 in zip(s1.frames, s2.frames):
            combined_frame = np.hstack((f1.numpy(), f2.numpy()))
            combined_frames.append(combined_frame)

        # Show the combined video
        for frame in combined_frames:
            cv2.imshow("Segment Pair", frame)
            key = cv2.waitKey(1000 // 30)  # Assuming 30 FPS
            if key != -1:
                break

        cv2.destroyAllWindows()

        # Get the user's preference
        while True:
            pref = input(
                "Enter 1 for the first segment or 2 for the second segment: "
            ).strip()
            if pref == "1":
                return [1.0, 0.0]
            elif pref == "2":
                return [0.0, 1.0]
            else:
                print("Invalid input. Please enter 1 or 2.")
