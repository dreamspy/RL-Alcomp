import gym
import numpy as np
from agents.DeepQLearningAgent import DeepQLearningAgent as currentAgent

environmentName = "CartPole-v1"
saveName = "DQA_2denseLayers_16nodesPerLayer"
renderEnvironment = False
onlyAcceptBetterModels = True

# Per batch we train for <trainingsPerBatch>, then evaluate the current agent by
# running <runsPerEvaluation> greedy simulations. This all done <nrOfBatches times.
# nrOfBatches = 200
# trainingsPerBatch = 10
# runsPerEvaluation = 100
nrOfBatches = 100
trainingsPerBatch = 10
runsPerEvaluation = 100

# Level 1 debugging
# db = False
db = True

# Level 2 debugging
# DB = False
DB = True

def db(head = "", tail = ""):
    if db:
        print str(head) + str(tail)

def DB(head = "", tail=""):
    if DB:
        print str(head) + str(tail)

def playGame():
    environment = gym.make(environmentName)
    # Change the agent by importing a different agent class
    agent = currentAgent(environment)

    averageReturns = np.zeros((nrOfBatches,2), dtype = int) # score for each evaluation
    highestAverage = 0

    for batch in range(nrOfBatches): # go through all train/evaluation batches
        DB("Batch nr: ", str(batch))
        DB("  Training...")
        for i in range(trainingsPerBatch):
            currentReturn = 0
            currentState = environment.reset()
            while True:
                if renderEnvironment: environment.render()
                action = agent.getNextAction(currentState)
                successorState, reward, done, info = environment.step(action)
                currentReturn += reward
                if done:
                    reward = 0
                    db("    Training Run: " + str(i) + ", exploration rate: " + str(agent.explorationRate) + " return: " + str(currentReturn), "")
                    agent.processLastStep(currentState, action, reward, done, successorState)
                    break
                agent.processLastStep(currentState, action, reward, done, successorState)
                currentState = successorState

        DB("  Evaluating agent...")
        returnSum = 0
        for i in range(runsPerEvaluation):
            currentReturn = 0
            currentState = environment.reset()
            while True:
                if i%20 == 0:
                    if renderEnvironment: environment.render()
                action = agent.getNextAction(currentState, greedy = True)
                currentState, reward, done, info = environment.step(action)
                currentReturn += reward
                if done:
                    # db("    Evaluation Run: " + str(r) + ", exploration rate: " + str(agent.explorationRate) + " return: " + str(currentReturn), "")
                    returnSum += currentReturn
                    break

        averageReturn = returnSum / runsPerEvaluation
        trainCount = (batch + 1) * trainingsPerBatch
        averageReturns[batch] = [trainCount, averageReturn]

        DB("  Average score over " + str(runsPerEvaluation) + " evaluation runs after " + str(trainCount) + " trainings = " + str(averageReturn) , "")

        if onlyAcceptBetterModels:
            if averageReturn > highestAverage:
                print "  Accepting new model"
                highestAverage = averageReturn
                agent.saveModel()
            else:
                print "  Rejecting new model"
                agent.loadModel()
        else:
            print "  Saving model"
            agent.saveModel()

    np.save(environmentName + "_" + saveName, averageReturns)
    loadTest = np.load(environmentName + "_" + saveName + ".npy")
    DB()
    DB("Final Results: [nr of training runs, average score]")
    DB(loadTest)
    DB("Saving final model")
    agent.saveModel()

if __name__ == "__main__":
    playGame()
