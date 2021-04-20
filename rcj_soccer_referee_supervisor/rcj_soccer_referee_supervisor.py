import logging
import os
import gym
import numpy as np
from time import sleep
from math import ceil
from datetime import datetime
from pathlib import Path, PosixPath

from referee.consts import DEFAULT_MATCH_TIME, TIME_STEP
from referee.event_handlers import JSONLoggerHandler, DrawMessageHandler
from referee.referee import RCJSoccerReferee




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
            3.56142833e-01, 5.62032341e-01, -4.99761261e-01,  4.47501229e-01,
            -5.14661245e-01, -4.99761261e-01,  4.22284228e-01, -2.39233253e-03,
            -4.99761261e-01, -2.93180141e-01, -5.35946988e-01,  4.99761261e-01,
            -3.23186305e-01,  5.31913584e-01,  4.99761261e-01, -1.25000000e-01,
            -4.74075686e-09,  4.99761261e-01,  0.00000000e+00,  0.00000000e+00
        ])

    def step(self, action):
        isdone = False
        self.referee.step(TIME_STEP)
        state = self.referee.emit_positions(action)
        #print(state)

        isdone = self.referee.tick()
        return state, 0, isdone, {}
    def render(self):
        pass

    def reset(self):

        self.referee.simulationResetPhysics()


        self.referee.reset_positions()
        self.referee.kickoff()
        

        return self.initial_state
    
env = ourEnv()

while 1:
    env.reset()
    SIM_STEPS = 1000
    print("NEW ENV")
    for i in range(SIM_STEPS):
        #print(i)
        sampl = np.random.uniform(low=-1.0, high=1.0, size=(6,))
        env.step(sampl)
    
    