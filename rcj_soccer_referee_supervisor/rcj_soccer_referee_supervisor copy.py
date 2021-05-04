import logging
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
from gym import spaces
import numpy as np
from time import sleep
from math import ceil, sqrt
from datetime import datetime
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import TD3

from referee.consts import DEFAULT_MATCH_TIME, TIME_STEP
from referee.event_handlers import JSONLoggerHandler, DrawMessageHandler
from referee.referee import RCJSoccerReferee

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor


from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

def output_path(
    directory: Path,
    team_blue_id: str,
    team_yellow_id: str,
    match_id: int,
    half_id: int,
) -> PosixPath:

    now_str = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    team_blue = team_blue_id.replace(' ', '_')
    team_yellow = team_yellow_id.replace(' ', '_')

    # Ensure the directory exists
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    filename = Path(f'{match_id}_-_{half_id}_-_{team_blue}_vs_{team_yellow}-{now_str}')
    return directory / filename


TEAM_YELLOW = os.environ.get("TEAM_YELLOW_NAME", "The Yellows")
TEAM_YELLOW_ID = os.environ.get("TEAM_YELLOW_ID", "The Yellows")
TEAM_YELLOW_INITIAL_SCORE = int(os.environ.get("TEAM_Y_INITIAL_SCORE", "0") or "0")
TEAM_BLUE = os.environ.get("TEAM_BLUE_NAME", "The Blues")
TEAM_BLUE_ID = os.environ.get("TEAM_BLUE_ID", "The Blues")
TEAM_BLUE_INITIAL_SCORE = int(os.environ.get("TEAM_B_INITIAL_SCORE", "0") or "0")
MATCH_ID = os.environ.get("MATCH_ID", 1)
HALF_ID = os.environ.get("HALF_ID", 1)
REC_FORMATS = [f for f in os.environ.get("REC_FORMATS", "").split(",") if f]
MATCH_TIME = int(os.environ.get("MATCH_TIME", DEFAULT_MATCH_TIME))

automatic_mode = True if "RCJ_SIM_AUTO_MODE" in os.environ.keys() else False

directory = Path('/out/') if automatic_mode else Path('reflog')
output_prefix = output_path(
    directory,
    TEAM_BLUE_ID,
    TEAM_YELLOW_ID,
    MATCH_ID,
    HALF_ID,
)
reflog_path = output_prefix.with_suffix('.jsonl')

class ourEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ourEnv, self).__init__()
        self.referee = referee = RCJSoccerReferee(
            match_time=MATCH_TIME,
            progress_check_steps=ceil(15/(TIME_STEP/1000.0)),
            progress_check_threshold=0.5,
            ball_progress_check_steps=ceil(10/(TIME_STEP/1000.0)),
            ball_progress_check_threshold=0.5,
            team_name_blue=TEAM_BLUE,
            team_name_yellow=TEAM_YELLOW,
            initial_score_blue=TEAM_BLUE_INITIAL_SCORE,
            initial_score_yellow=TEAM_YELLOW_INITIAL_SCORE,
            penalty_area_allowed_time=15,
            penalty_area_reset_after=2,
        )
        #since reset starts the same position, and I'm too lazy to provide a new observation...
        self.initial_state = np.array([ 
            3.56142833e-01, 5.62032341e-01,  math.sin(-4.99761261e-01), math.cos(-4.99761261e-01),   4.47501229e-01,
            -5.14661245e-01, math.sin(-4.99761261e-01), math.cos(-4.99761261e-01),  4.22284228e-01, -2.39233253e-03,
            math.sin(-4.99761261e-01), math.cos(-4.99761261e-01), -2.93180141e-01, -5.35946988e-01,  math.sin(-4.99761261e-01), math.cos(-4.99761261e-01),
            -3.23186305e-01,  5.31913584e-01,  math.sin(-4.99761261e-01), math.cos(-4.99761261e-01), -1.25000000e-01,
            -4.74075686e-09,  math.sin(-4.99761261e-01), math.cos(-4.99761261e-01),  0.00000000e+00,  0.00000000e+00, 0,0
        ])
        self.action_space = spaces.Box(low=-1, high=1, shape=(6, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(28, ), dtype=np.float32)
        self.SIM_STEPS = 1000
        self.score_blue = 0
        self.score_yellow = 0

    def step(self, action):
        isdone = False
        self.referee.step(TIME_STEP)
        state = self.referee.emit_positions(np.clip(action, -1, 1))
        self.SIM_STEPS-= 1
        isdone = self.referee.tick()
        
        return state, self.get_reward(), self.SIM_STEPS<1, {}
    
    def get_reward(self):
        reward = 0
        
        if self.referee.score_blue > self.score_blue:
            reward = 1
            self.score_blue = self.referee.score_blue

        

        elif self.referee.score_yellow > self.score_yellow:
            self.score_yellow = self.referee.score_yellow
            reward = -1
        
        ball = self.referee.ball_translation
        x, y = ball[0] / 0.8 , ball[2] / 0.65
        dst1 = sqrt((x-1)**2 + (y)**2)
        dst2 = sqrt((x+1)**2 + (y)**2)
        dst_offset = (dst1-dst2)*0.01
        return reward

    def render(self):
        pass

    def reset(self):
        
        self.SIM_STEPS = 1000
        #self.referee.simulationReset()
        #self.referee.simulationResetPhysics()


        self.referee.reset_positions()
        self.referee.kickoff()
        

        return self.initial_state

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

log_dir = r"C:\Users\SkittishHardware\Documents\AIFootball\rcj-soccer-sim-communication - Copy\controllers\rcj_soccer_referee_supervisor"

env = ourEnv()
#env = Monitor(env, log_dir)



n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=100000)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000000)
plot_results("tmp/")

