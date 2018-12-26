import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys

class randomAgent:

    def __init__(self, env):
        # Don't change these, neccessary for all agents
        self.environment = env
        self.obervationSpaceShape = env.observation_space.shape[0]
        self.actionSpaceShape = env.action_space.n

        # Settings for current agent

    # Return next action from agent
    def getNextAction(self, currentState, greedy = False):
        return self.environment.action_space.sample()


    # Update policy of agent with sample returns
    def updatePolicy(self, currentState, lastAction, reward, done, successorState):
        pass

    # Save current policy model
    def saveModel(self, modelName = "kerasModel"):
        pass

    # Load mode
    def loadModel(self, modelName = "kerasModel"):
        pass

