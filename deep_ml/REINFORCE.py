import numpy as np

def compute_policy_gradient(theta: np.ndarray, episodes: list[list[tuple[int, int, float]]]) -> np.ndarray:
    """
    Estimate the policy gradient using REINFORCE.

    Args:
        theta: (num_states x num_actions) policy parameters.
        episodes: List of episodes, where each episode is a list of (state, action, reward).

    Returns:
        Average policy gradient (same shape as theta).
    """
    grad_list = []
    softmax_theta = theta - np.max(theta, axis=1)
    softmax_theta = np.exp(softmax_theta) / np.exp(softmax_theta).sum(axis=1)
    #import pdb; pdb.set_trace()

    for episode in episodes:
        r_seq = [r for s, a, r in episode]
        g_t = np.cumsum(r_seq[::-1])[::-1]
        grad_episode = np.zeros_like(theta)
        for t, (s, a, r) in enumerate(episode):
            grad_episode[s, :] += -softmax_theta[s, :] * g_t[t]
            #grad_episode[s, a] += (1 - softmax_theta[s, a]) * g_t[t]    # [s, a] has been updated already
            grad_episode[s, a] += 1 * g_t[t]
        grad_list.append(grad_episode)
    return np.mean(np.stack(grad_list, axis=0), axis=0)


theta = np.zeros((2,2))
episodes = [[(0,1,0), (1,0,1)], [(0,0,0)]]

print(compute_policy_gradient(theta, episodes))