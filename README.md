# FrozenLakeRL
FrozenLake Environment from openai gym

The problem is being solved using q-learning algorithm with q-table.
The agent should find a safe path across frozen lake with holes in it. Agent is trained for some amount of episodes, falling into the hole (H) results in an end of an episode and achieving the goal tile results in an end of an episode and a reward of 1 point.
The overall goal is to maximize the reward over time, or to find the optimal policy with which the agent can go from the start point to the end.

Q-learning algorithm fills a q-table with expected future rewards for (state, action) pairs, that is, if an agent will take an action from current state, he will get reward written in the table.
When the table is done (converged to some values), the agent can achieve a goal with a greedy algorithm - at every step, being in the current state, it takes an action with maximum expected reward until it reaches the goal and receives the reward.

Q-learning algorithm uses epsilon-greedy policy. This means that the agent takes random action with probability epsilon and greedily takes an action using q-table with probability (1 - epsilon), and epsilon decreases over time to 0. It is proven, that if the epsilon decreases to zero with number of episodes increasing to infinity, the algorithm converges to the optimal policy, that is, an agent learns to achieve the goal.