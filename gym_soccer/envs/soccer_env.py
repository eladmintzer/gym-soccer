import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
MAX_TIMESTAMP=600000
MIN_COORDINATE_FIELD = -55.0
MAX_COORDINATE_FIELD = 55.0
MAX_POWER = 100
MIN_POWER = 0
MAX_DISTANCE = 113.0
NUM_OF_PLAYER_WITHOUT_GOALI = 10
NUM_OF_PLAYER_TOTAL = 11
NUMR_OF_TEAM = 2
NUM_OF_OBJ = 3 # left/right goal ball
NUM_OF_OBJ_ID = NUM_OF_OBJ + 1 # NUM_OF_OBJ + N/A
NUM_OF_FOUL_OPTION = 3
NUM_OF_DASH_OPTION = 1
NUM_OF_WIDTH_OPTION = 4
NUM_OF_QUALITY_OPTION = 3
import logging
logger = logging.getLogger(__name__)


class FullSoccerEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        self.server_process = None
        self.server_port = None
        self._configure_environment()

        self.action_space = spaces.Tuple((spaces.Discrete(3),
        # KICK
        spaces.Box(low=MIN_POWER, high=MAX_POWER, shape=1),   # kick p1 = > power (float)
        spaces.Box(low=0, high=360, shape=1),                 # kick p2 = > direction(float)
        # move
        spaces.Box(low=MIN_COORDINATE_FIELD, high=MAX_COORDINATE_FIELD, shape=2), # move p1/2 = > x/y(float)
        #tackle
        spaces.Box(low=0, high=360, shape=1),                 # tackle p1 = > PowerOrAngle(float) TODO power or angle
        spaces.Discrete(NUM_OF_FOUL_OPTION),                  # tackle p2 = > foul(Str / None)
        # dash
        spaces.Box(low=MIN_POWER, high=MAX_POWER, shape=1),   # dash p1 = > power (float)
        spaces.Discrete(NUM_OF_FOUL_OPTION),                  # dash p2 = > dash(float / None) TODO
        # turn_neck
        spaces.Box(low=0, high=360, shape=1),   # turn_neck p1 = > degree(float)
        # turn
        spaces.Box(low=0, high=360, shape=1),   # turn p1 = > degree(float)
        # point_to
        spaces.Discrete(2), # point_to p1 = > off(boolean)
        spaces.Box(low=0, high=MAX_DISTANCE, shape=1), # point_to p2 = > distance(float / None)
        spaces.Box(low=0, high=360, shape=1), # point_to p3 = > direction(float / None)
        # say p1 = > message(str) TODO: no param just act
        # change_view
        spaces.Discrete(NUM_OF_WIDTH_OPTION),# change_view p1 = > width(str)
        spaces.Discrete(NUM_OF_QUALITY_OPTION),# change_view p2 = > quality(str / None)
        # attention_to
        spaces.Discrete(2), # attention_to p1 = > off(boolean)
        spaces.Discrete(3), # attention_to p2 = > team(str / None)
        spaces.Discrete(NUM_OF_PLAYER_TOTAL + 1)))# attention_to p3 = > player_num(int / None)

        self.observation_space = spaces.Tuple((spaces.Discrete(MAX_TIMESTAMP),  # timeStamp
           #spaces.Tuple((                 # position(
           spaces.Box(low=MIN_COORDINATE_FIELD, high=MAX_COORDINATE_FIELD, shape=2),  # x,y
           spaces.Box(low=-180.0, high=180.0, shape=1),  # pin
           #)),                            # position)
           # spaces.Tuple(  # velocity(
           spaces.Box(low=-1, high=2.0, shape=2),  # vx,vy
           # ),  # position)
           # spaces.Tuple(  # velocity(
           spaces.Discrete(10),  # robocup_agent_type
           spaces.Discrete(10),  #  "robocup_agent_state": 1,
           spaces.Box(low=-180.0, high=180.0, shape=1),  #  "neck_angle": -81,
           spaces.Discrete(2),  #  "view_quality": "h"/"l"
           spaces.Box(low=0, high=180.0, shape=1),  #  "view_angle": 120,
           spaces.Box(low=0, high=10000, shape=1),  #  "agent_stamina": 8000,
           spaces.Box(low=-1, high=1, shape=1),  #  "agent_stamina_effort": 0.994703,
           spaces.Box(low=-1, high=1, shape=1),  #  "agent_stamin_rec": 1,
           spaces.Box(low=0, high=200000, shape=1),  #  "agent_stamin_total": 130600,
           spaces.Discrete(2),  #  "robocup_focus_side": "l",
           spaces.Discrete(11),  #  "robocup_focus_num": 1,
           # ViewewAgent all of this spaces in shape of (2 team,11 players,? field_shape) total 2*11*? = 22?
           spaces.Box(low=0, high=1, shape=(2,11,2)),  # visible_agent_{id,team}_prob": 0-1
           spaces.Box(low=0, high=360, shape=(2,11)),  # ViewewAgent relative_angle 0-360
           spaces.Box(low=0, high=60, shape=(2,11)),  # ViewewAgent dist 0-60
           spaces.Box(low=MIN_COORDINATE_FIELD, high=MAX_COORDINATE_FIELD, shape=(2,11,2)), # viewed_agent x,y
           spaces.Box(low=-180.0, high=180.0, shape=(2,11)), # viewed_agent pin
           # ViewewObjects all of this spaces in shape of (2 team,11 players,? field_shape) total 2*11*? = 22?
           spaces.Box(low=0, high=1, shape=(NUM_OF_OBJ)),  # ViewewObjects visible_object_{id,type}_prob": 0-1
           spaces.Box(low=0, high=360, shape=(NUM_OF_OBJ)),  # ViewewObjects relative_angle 0-360
           spaces.Box(low=0, high=60, shape=(NUM_OF_OBJ)),  # ViewewObjects dist 0-60
           spaces.Box(low=MIN_COORDINATE_FIELD, high=MAX_COORDINATE_FIELD, shape=(NUM_OF_OBJ, 2)),  # ViewewObjects x,y
           spaces.Discrete(NUM_OF_OBJ_ID)))  # ViewewObjects ID

    def _step(self, action):
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        return ob, reward, {}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        # action_type = ACTION_LOOKUP[action[0]]
        # if action_type == hfo_py.DASH:
        #     self.env.act(action_type, action[1], action[2])
        # elif action_type == hfo_py.TURN:
        #     self.env.act(action_type, action[3])
        # elif action_type == hfo_py.KICK:
        #     self.env.act(action_type, action[4], action[5])
        # else:
        #     print('Unrecognized action %d' % action_type)
        #     self.env.act(hfo_py.NOOP)

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        # if self.status == hfo_py.GOAL:
        #     return 1
        # else:
        #     return 0

    def _reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        # while self.status == hfo_py.IN_GAME:
        #     self.env.act(hfo_py.NOOP)
        #     self.status = self.env.step()
        # while self.status != hfo_py.IN_GAME:
        #     self.env.act(hfo_py.NOOP)
        #     self.status = self.env.step()
        # return self.env.getState()

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        # if close:
        #     if self.viewer is not None:
        #         os.kill(self.viewer.pid, signal.SIGKILL)
        # else:
        #     if self.viewer is None:
        #         self._start_viewer()