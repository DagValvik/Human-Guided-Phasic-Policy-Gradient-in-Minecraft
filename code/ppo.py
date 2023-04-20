import numpy as np
import torch as th


def value_update(states, value_targets, policy, optimizer, gamma):
    value_losses = []
    for state, value_target in zip(states, value_targets):
        _, value_prediction, _ = policy.get_output_for_observation(state, policy.initial_state(1), dummy_first)
        value_losses.append((value_prediction.squeeze() - value_target) ** 2)
    
    value_loss = th.mean(th.stack(value_losses))
    
    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()



# Functions for PPO update, GAE, etc.
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[
            rand_ids, :
        ], returns[rand_ids, :], advantages[rand_ids, :]


def ppo_update(
    ppo_epochs,
    mini_batch_size,
    states,
    actions,
    log_probs,
    returns,
    advantages,
    policy,
    optimizer,
    ppo_clip=0.2,
    value_coef=0.5,
    max_grad_norm=0.5,
    entropy_coef=0.01,
):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(
            mini_batch_size, states, actions, log_probs, returns, advantages
        ):
            pi_distribution, values, _ = policy.get_output_for_observation(
                state
            )
            new_log_probs = policy.get_logprob_of_action(
                pi_distribution, action
            )

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = th.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantage

            actor_loss = -th.min(surr1, surr2).mean()
            critic_loss = (return_ - values).pow(2).mean()

            entropy = policy.get_entropy(pi_distribution).mean()

            loss = (
                actor_loss + value_coef * critic_loss - entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()


def compute_gae(rewards, values, masks, gamma, lam):
    """
    Calculate the generalized advantage estimate
    """
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        next_value = values[step + 1] if step < len(rewards) - 1 else 0
        delta = rewards[step] + gamma * \
            next_value * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
