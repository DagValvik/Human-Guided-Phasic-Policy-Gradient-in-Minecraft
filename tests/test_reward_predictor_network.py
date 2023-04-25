import queue
from code.reward_predict import RewardPredictorCore, RewardPredictorNetwork
from code.segment import Segment

import torch


# Test if the RewardPredictorCore can be created
def test_reward_predictor_core_creation():
    core = RewardPredictorCore()
    assert isinstance(core, RewardPredictorCore)


# Test if the RewardPredictorNetwork can be created
def test_reward_predictor_network_creation():
    network = RewardPredictorNetwork()
    assert isinstance(network, RewardPredictorNetwork)


# Test if the RewardPredictorNetwork can calculate rewards
def test_reward_calculation():
    network = RewardPredictorNetwork()
    obs = torch.randn(1, 3, 128, 128)  # Create a random observation
    assert obs.size() == torch.Size([1, 3, 128, 128])
    rewards = network.reward(obs)
    assert rewards.size() == torch.Size([1])
    # Check if the reward is a float
    assert isinstance(rewards.item(), float)


# Test if the RewardPredictorNetwork can perform a single training step
def test_train_step():
    network = RewardPredictorNetwork()
    # Create two segments of video frames
    s1 = Segment()
    s2 = Segment()
    # Add a frame to each segment
    s1.add_frame(torch.randn(1, 3, 128, 128), 0.0)
    s1.finish()
    s2.add_frame(torch.randn(1, 3, 128, 128), 0.5)
    s2.finish()
    # Create a preference list
    pref = [1.0, 0.0]
    # Perform a training step
    loss = network.train_step(s1, s2, pref)
    assert isinstance(loss, float)
    assert loss > 0.0  # The loss should be positive


# Test if the RewardPredictorNetwork can train on a preference queue of segment pairs
def test_train():
    # Create queue
    pref_queue = queue.Queue()
    network = RewardPredictorNetwork(pref_queue=pref_queue)
    # Create two segments of video frames
    s1 = Segment()
    s2 = Segment()
    # Add a frame to each segment
    s1.add_frame(torch.randn(1, 3, 128, 128), 0.0)
    s1.finish()
    s2.add_frame(torch.randn(1, 3, 128, 128), 0.0)
    s2.finish()

    pref_queue.put((s1, s2, [1.0, 0.0]))

    # Train on the preference queue
    network.train(max_iterations=1)

    # Check that quque is empty
    assert pref_queue.empty()


# Test if the RewardPredictorNetwork can save and load its parameters
def test_save_and_load():
    network = RewardPredictorNetwork()
    save_path = "tests/test_weights.pth"
    network.save(save_path)

    network2 = RewardPredictorNetwork()
    network2.load(save_path)

    # Check if the weights are the same after loading
    for p1, p2 in zip(network.core.parameters(), network2.core.parameters()):
        assert torch.allclose(p1, p2)
