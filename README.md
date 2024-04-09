# TVC_RL 

Notes about files:

`utils.py`: Contains `run_episode(policy: function, render=False)` function that takes as parameter `policy(observation: List)` that runs an episode and returns the cumulative reward

`baseline.py`: Contains a baseline policy that contains two sub-policies: `slow_descent`, which makes the vehicle descend at a given y velocity, and `orient_vehicle`, which attempts to make the vehicle perpendicular to the ground.

`evaluate_policy.py`: Used for evaluating policies. Calculates the average cumulative reward over many episodes.


Env Notes:
 - State space: [position_y, velocity_y, velocity_x, angular_velocity, angle]
 - Action space: [left, mid, right, none]