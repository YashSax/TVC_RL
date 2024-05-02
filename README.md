# TVC_RL 
## Instructions to Run:

Even though there's a local copy of spinup installed, there are additional dependencies needed. Follow the instructions [here](https://spinningup.openai.com/en/latest/user/installation.html) to install the needed dependencies. Additionally, run `pip install -r requirements.txt`.

`baseline_evalute.py`: Runs the baseline 300 times and prints the mean, min, and max return as well as a breakdown of the different types of landing violations (angle, angular velocity, velocity, fuel)

`spintest.py`: Running `spinpg()` will train a PPO agent on the TVC environment. Pass in `safeRL="teacher_assist"` or `safeRL="automated_recovery"` to `spinpg` to run Teacher Assistance or Automated Recovery, respectively.

## Additional Files and Folders:

`final_models`: Contains the trained models for Vanilla PPO, Teacher Assistance, and Hardcoded Recovery

`utils.py`: Contains various utility functions, such as `run_episode(policy)`, `evaluate_policy(policy)`, `crashed(policy)`, `is_safe(env)`, and `is_dangerous(env)`. 


`baseline.py`: Contains a baseline policy that contains two sub-policies: `slow_descent`, which makes the vehicle descend at a given y velocity, and `orient_vehicle`, which attempts to make the vehicle perpendicular to the ground.

`visualize_results.ipynb`: Code used to generate graphs in report.

Env Notes:
 - State space: [position_y, velocity_y, velocity_x, angular_velocity, angle]
 - Action space: [left, mid, right, none]

Safe RL Notes:
 - Potential safety restrictions:
    - Sometimes the rocket can just stay in the air forever until the env times out. Make sure that the rocket lands or it'll run out of fuel.
    - Rocket can stand only so much y_vel or x_vel or angular_vel on landing