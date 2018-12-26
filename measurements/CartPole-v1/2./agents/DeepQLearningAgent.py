import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import sys

class DeepQLearningAgent:
    
    def __init__(self, env):
        self.obervationSpaceShape = env.observation_space.shape[0]
        self.actionSpaceShape = env.action_space.n
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
        self.model = self.makeNeuralNet()

    def makeNeuralNet(self):
        model = Sequential()
        for i in range(len(self.nnDenseLayersSettings)):
            nodes = self.nnDenseLayersSettings[i][0]
            activation = self.nnDenseLayersSettings[i][1]
            if i == 0:
                model.add(Dense(nodes,
                                input_shape = (self.obervationSpaceShape,),
                                activation = activation))
            else:
                model.add(Dense(nodes,
                                activation = activation))
        model.add(Dense(self.actionSpaceShape,
                        activation = "linear"))
        model.compile(loss = "mse",
                      optimizer = Adam(lr = self.learningRate))
        return model

    def getNextAction(self, currentState, greedy = False):
        currentState = np.array ([currentState])
        if np.random.rand() <= self.explorationRate and not greedy:
            return random.randrange(self.actionSpaceShape)
        currentQ = self.model.predict(currentState)
        return np.argmax(currentQ[0])

    def processLastStep(self, currentState, lastAction, reward, done, successorState):
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
            if done:
                newQ_a = reward
            else:
                newQ_a = (reward + self.gamma * np.amax(self.model.predict(successorState)))
            newQ = self.model.predict(currentState)
            newQ[0][lastAction] = newQ_a
            self.model.fit(currentState, newQ, verbose = 0)

    def saveModel(self, text = "kerasModel"):
        self.model.save_weights(str(text) + '.pth.tar')

    def loadModel(self, text = "kerasModel"):
        self.model.load_weights(str(text) + '.pth.tar')

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
