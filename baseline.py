from rocketgym_local.environment import Environment
from utils import *
import math

# If you're already vertical
MID_DESCENT_VEL = -4  # How fast should you descend during flight
LANDING_DESCENT_VEL = -1  # How fast should you descend when you're about to land
LANDING_POS_THRESH = 0.5  # What's the threshold for being "about to land"

# If you're not vertical
P = 1
D = 0.2
ANGLE_THRESH_UPPER = 10 / 180 * math.pi
ANGLE_THRESH_LOWER = -10 / 180 * math.pi


def slow_descent(observation):
    pos_y, vel_y, _, _, _ = observation
    if pos_y < LANDING_POS_THRESH:
        return ACTION_NONE if vel_y > LANDING_DESCENT_VEL else ACTION_MID
    return ACTION_NONE if vel_y > MID_DESCENT_VEL else ACTION_MID


def orient_vehicle(observation):
    _, _, _, ang_vel, angle = observation
    # print("Angle:", angle * 180 / math.pi)
    return ACTION_RIGHT if (P * angle - D * ang_vel) > 0 else ACTION_LEFT


def rocket_off_center(observation):
    _, _, _, _, angle = observation
    return angle > ANGLE_THRESH_UPPER or angle < ANGLE_THRESH_LOWER


def baseline_policy(observation):
    if rocket_off_center(observation):
        action = orient_vehicle(observation)
    else:
        action = slow_descent(observation)
    return action

if __name__ == "__main__":
    while True:
        print(run_episode(baseline_policy))
