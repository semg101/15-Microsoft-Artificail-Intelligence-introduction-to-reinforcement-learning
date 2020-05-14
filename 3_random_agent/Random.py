#Now that you have examine the SimpleRoomsEnv and CliffWalkingEnv environments in the Environments.py file, 
#let's play around with an agent in those environments.
import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../") 

'''
from lib.envs.simple_rooms import SimpleRoomsEnv
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.simulation import Experiment
'''


from simple_rooms import SimpleRoomsEnv
from cliff_walking import CliffWalkingEnv
from simulation import Experiment


#Below is the agent interface. At the minimum, an agent will have an act() function that takes an observation, and return an action.
class Agent(object):

    def __init__(self, actions):
        self.actions = actions
        self.num_actions = len(actions)

    def act(self, obs):
        raise NotImplementedError


#We've given you the implementation of a random agent. It's act() function will just return a random action within the valid action space.
class RandomAgent(Agent):
    
    def __init__(self, actions):
        super(RandomAgent, self).__init__(actions)
    
    def act(self, obs):
        return np.random.randint(0, self.num_actions)




#Now let's run the experiment. We'll start with the SimpleRoomsEnv environment, with just 5 episodes.
interactive = True

max_number_of_episodes = 5
env = SimpleRoomsEnv()
agent = RandomAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_agent(max_number_of_episodes, interactive)

#Let's do one with the CliffWalkingEnv environment. This time with 10 episodes.
''' #it does not work
max_number_of_episodes = 10
env = CliffWalkingEnv()
agent = RandomAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_agent(max_number_of_episodes, interactive)
'''

#Let's set interactive to False, and run the experiments again. This time with 100 episodes each.
interactive = False

max_number_of_episodes = 100
env = SimpleRoomsEnv()
agent = RandomAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_agent(max_number_of_episodes, interactive)

''' #it does not work
max_number_of_episodes = 100
env = CliffWalkingEnv()
agent = RandomAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_agent(max_number_of_episodes, interactive)
'''
