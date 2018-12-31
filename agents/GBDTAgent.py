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
        
        self.explorationRate = 1.0
        self.explorationFinalValue = 0.01
        self.explorationDiscountFactor = 0.96
        self.gamma = 0.95
        self.learningRate = 0.001
        self.bufferSize = 1000
        self.buffer = deque(maxlen=self.bufferSize)
        self.batchSize = 20
        self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=100, n_jobs=-1))

        self.ready = False

    # Return next action from agent
    def getNextAction(self, currentState, greedy = False):
        if np.random.rand() < self.explorationRate and not greedy:
            return random.randrange(self.actionSpaceShape)
        if self.ready:
            currentQ = self.model.predict(currentState)
        else:
            currentQ = np.zeros(self.actionSpaceShape).reshape(1, -1)
        return np.argmax(currentQ[0])
      
    def mem(self, currentState, lastAction, reward, done, successorState):
        self.buffer.append((currentState, lastAction, -reward if done else reward, successorState, done))

    # Update policy of agent with sample returns
    def updatePolicy(self):
        # Update exploration rate at end of each epoch

        if len(self.buffer) < self.batchSize:
            return
        batch = random.sample(self.buffer, int(len(self.buffer)/1))
        
        X = []
        targets = []
        for currentState, action, reward, successorState, terminal in batch:
            q_update = reward
            if not terminal:
                if self.ready:
                    q_update = (reward + self.gamma * np.amax(self.model.predict(successorState)))
                    #print(self.model.predict(state_next))
                else:
                    q_update = reward
            if self.ready:
                q_values = self.model.predict(currentState)
            else:
                q_values = np.zeros(self.actionSpaceShape).reshape(1, -1)
            q_values[0][action] = q_update
            
            #print(state)
            #print(action)
            #print(q_values)
            X.append(list(currentState[0]))
            targets.append(q_values[0])
        #print(X)
        #print(targets)
        self.model.fit(X, targets)
        self.ready = True
        
        self.explorationRate *= self.explorationDiscountFactor
        self.explorationRate = max(self.explorationFinalValue, self.explorationRate)

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