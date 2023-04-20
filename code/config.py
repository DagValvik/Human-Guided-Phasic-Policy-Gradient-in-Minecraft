import json
import os

# DO NOT CHANGE THESE VALUES
# These will be enforced in the evaluation server:
# if you tinker with them, your submissions will almost certainly fail!

# 10 for evaluation + 1 leaderboard video
EVAL_EPISODES = int(os.getenv("AICROWD_NUM_EVAL_EPISODES", 10))
# This is only used to limit steps when debugging is on.
# Environments will automatically return done=True once
# the environment-specific timeout is reached
EVAL_MAX_STEPS = int(os.getenv("AICROWD_NUM_EVAL_MAX_STEPS", 1e9))
