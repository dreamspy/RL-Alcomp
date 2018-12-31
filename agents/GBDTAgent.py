import random
import numpy as np
from collections import deque

from joblib import dump, load
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

import sys

class GBDTAgent:

    def __init__(self, env):
        # Don't change these, neccessary for all agents
        self.environment = env
        self.obervationSpaceShape = env.observation_space.shape[0]
        self.actionSpaceShape = env.action_space.n

        # Settings for current agent
        self.nnDenseLayersSettings = [[20, "relu"],
                                      [20, "relu"]]
        self.explorationRate = 1.0
        self.explorationFinalValue = 0.01
        self.explorationDiscountFactor = (0.01)**(1/400.) #0.994260074
        self.gamma = 0.95
        self.learningRate = 0.001
        self.bufferSize = 1000000
        self.buffer = deque(maxlen=self.bufferSize)
        self.batchSize = 30
        self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))

        self.ready = False

    # Return next action from agent
    def getNextAction(self, currentState, greedy = False):
        currentState = np.array ([currentState])
        if not self.ready or np.random.rand() <= self.explorationRate and not greedy:
            return random.randrange(self.actionSpaceShape)
        currentQ = self.model.predict(currentState)
        return np.argmax(currentQ[0])

    # Update policy of agent with sample returns
    def updatePolicy(self, currentState, lastAction, reward, done, successorState):
        # Save last state action pair to buffer
        self.buffer.append((currentState, lastAction, reward, successorState, done))

        # Update exploration rate at end of each epoch
        if done:
            if self.explorationRate > self.explorationFinalValue:
                self.explorationRate *= self.explorationDiscountFactor
            elif self.explorationRate < self. explorationFinalValue:
                self.explorationRate = self.explorationFinalValue

        if len(self.buffer) < self.batchSize:
            return
        batch = random.sample(self.buffer, self.batchSize)
        for currentState, lastAction, reward, successorState, done in batch:
            successorState = np.array([successorState])
            currentState = np.array([currentState])
            if not self.ready or done:
                newQ_a = reward
            else:
                newQ_a = (reward + self.gamma * np.amax(self.model.predict(successorState)))
            newQ = self.model.predict(currentState) if self.ready else np.zeros(self.actionSpaceShape)
            newQ[0][lastAction] = newQ_a
            self.model.fit(currentState, newQ)
            self.ready = True

    # Save current policy model
    def saveModel(self, modelName ="kerasModel"):
        dump(self.model, f'{modelName}.joblib.gz')

    # Load mode
    def loadModel(self, modelName ="kerasModel"):
        self.model = load(f'{modelName}.joblib.gz')

    # def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    #     filepath = os.path.join(folder, filename)
    #     if not os.path.exists(folder):
    #         print("Checkpoint Directory does not exist! Making directory {}".format(folder))
    #         os.mkdir(folder)
    #     else:
    #         print("Checkpoint Directory exists! ")
    #     if self.saver == None:
    #         self.saver = tf.train.Saver(self.nnet.graph.get_collection('variables'))
    #     with self.nnet.graph.as_default():
    #         self.saver.save(self.sess, filepath)
    #
    # def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
    #     filepath = os.path.join(folder, filename)
    #     if not os.path.exists(filepath + '.meta'):
    #         raise("No model in path {}".format(filepath))
    #     with self.nnet.graph.as_default():
    #         self.saver = tf.train.Saver()
    #         self.saver.restore(self.sess, filepath)