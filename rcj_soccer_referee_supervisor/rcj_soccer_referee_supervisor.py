import logging
import os
import gym
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

# referee = RCJSoccerReferee(
#     match_time=MATCH_TIME,
#     progress_check_steps=ceil(15/(TIME_STEP/1000.0)),
#     progress_check_threshold=0.5,
#     ball_progress_check_steps=ceil(10/(TIME_STEP/1000.0)),
#     ball_progress_check_threshold=0.5,
#     team_name_blue=TEAM_BLUE,
#     team_name_yellow=TEAM_YELLOW,
#     initial_score_blue=TEAM_BLUE_INITIAL_SCORE,
#     initial_score_yellow=TEAM_YELLOW_INITIAL_SCORE,
#     penalty_area_allowed_time=15,
#     penalty_area_reset_after=2,
# )

# if automatic_mode:
#     referee.simulationSetMode(referee.SIMULATION_MODE_FAST)
#     for recorder in recorders:
#         recorder.start_recording()

# referee.add_event_subscriber(JSONLoggerHandler(reflog_path))
# referee.add_event_subscriber(DrawMessageHandler())

# referee.kickoff()

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
    def step(self, action):
        isdone = False
        self.referee.step(TIME_STEP)
        self.referee.emit_positions(actions)
        #print(self.referee.score_yellow, self.referee.score_blue)
        isdone = self.referee.tick()
        return object(), 0, isdone, {}
    def render(self):
        pass

    def reset(self):
        self.referee.simulationResetPhysics()
        #self.referee.add_event_subscriber(JSONLoggerHandler(reflog_path))
        #self.referee.add_event_subscriber(DrawMessageHandler())
        self.referee.reset_positions()
        self.referee.kickoff()
        
        return object()
    
env = ourEnv()

while 1:
    env.reset()
    SIM_STEPS = 100000

    for i in range(SIM_STEPS):
        #print(i)
        env.step(0)
    
    

# # The "event" loop for the referee
# while referee.step(TIME_STEP) != -1:
#     referee.emit_positions()
    
#     # If the tick does not return True, the match has ended and the event loop
#     # can stop
#     if not referee.tick():
#         break

# # When end of match, pause simulator immediately
# referee.simulationSetMode(referee.SIMULATION_MODE_PAUSE)

