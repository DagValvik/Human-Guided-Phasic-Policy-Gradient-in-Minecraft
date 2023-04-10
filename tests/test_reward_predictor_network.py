from code.reward_predict import RewardPredictorCore, RewardPredictorNetwork

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
    obs = torch.randn(1, 3, 360, 640)  # Random tensor simulating a video frame
    rewards = network.reward(obs)
    assert rewards.size() == torch.Size([1])


# Test if the RewardPredictorNetwork can perform a single training step
def test_train_step():
    network = RewardPredictorNetwork()
    s1 = torch.randn(1, 3, 360, 640)
    s2 = torch.randn(1, 3, 360, 640)
    pref = [1.0, 0.0]
    loss = network.train_step(s1, s2, pref)
    assert isinstance(loss, float)


# Test if the RewardPredictorNetwork can save and load its parameters
def test_save_and_load():
    network = RewardPredictorNetwork()
    save_path = "test_weights.pth"
    network.save(save_path)

    network2 = RewardPredictorNetwork()
    network2.load(save_path)

    # Check if the weights are the same after loading
    for p1, p2 in zip(network.core.parameters(), network2.core.parameters()):
        assert torch.allclose(p1, p2)
