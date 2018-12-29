import gym
import numpy as np
from agents.DeepQLearningAgent import DeepQLearningAgent as currentAgent
import sys

# Per batch we train for <trainingsPerBatch>, then evaluate the current agent by
# running <runsPerEvaluation> greedy simulations. This all done <nrOfBatches times.

# #Measure how many trainings on average do we need to solve the environment
# for nrOfEvaluations:
#     #Alternate between learning and evaluating a policy
#     for nrOfBatches:
#         for trainingsPerBatch:
#         for runsPerEvaluation:

environmentName = "CartPole-v1"
saveName = "CartPole_DQA_2x20"
if len(sys.argv) > 2:
    saveName += "_" + str(sys.argv[1])
renderEnvironment = False
onlyAcceptBetterModels = False
onlySaveBestModel = True
saveToArgDir = True

nrOfEvaluations = 10
nrOfBatches = 100
trainingsPerBatch = 5
runsPerEvaluation = 100
# envSolvedAtScore = 195

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

def createSaveName(name = False, evaluation = -1, batch = -1, text = None):
    tempText = ""
    if len(sys.argv) > 2:
        tempText += str(sys.argv[2]) + "/"
    if name:
        tempText += saveName + "_"
    if evaluation != -1:
        tempText += "eval-" + str(evaluation) + "_"
    if batch != -1:
        temptext += "batch-" + str(batch) + "_"
    if text != None:
        tempText += text
    return tempText

def solveEnvironment():

    averageRunReturns = []
    environment = gym.make(environmentName)
    for evaluation in range(nrOfEvaluations):
        DB("Evaluation nr: " + str(evaluation))
        # Change the agent by importing a different agent class
        agent = currentAgent(environment)
        averageBatchReturns = np.zeros((nrOfBatches,2), dtype = int) # score for each batch
        highestAverage = 0
        for batch in range(nrOfBatches): # go through all train/evaluation batches
            DB("  Batch nr: ", str(batch))
            DB("    Exporation Rate: " + str(agent.explorationRate))
            DB("    Training...")
            for i in range(trainingsPerBatch):
                currentReturn = 0
                currentState = environment.reset()
                while True:
                    # if renderEnvironment: environment.render()
                    action = agent.getNextAction(currentState)
                    successorState, reward, done, info = environment.step(action)
                    currentReturn += reward
                    if done:
                        reward = 0
                        # db("      Training Run: " + str(i) + ", exploration rate: " + str(agent.explorationRate) + " return: " + str(currentReturn), "")
                        agent.processLastStep(currentState, action, reward, done, successorState)
                        break
                    agent.processLastStep(currentState, action, reward, done, successorState)
                    currentState = successorState

            DB("    Evaluating agent...")
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
                        # db("      Evaluation Run: " + str(r) + ", exploration rate: " + str(agent.explorationRate) + " return: " + str(currentReturn), "")
                        returnSum += currentReturn
                        break
                # currentState = environment.reset()

            averageReturn = returnSum / runsPerEvaluation
            trainCount = (batch + 1) * trainingsPerBatch
            averageBatchReturns[batch] = [trainCount, averageReturn]

            DB("    Average score over " + str(runsPerEvaluation) + " evaluation runs after " + str(trainCount) + " trainings = " + str(averageReturn) , "")

            if averageReturn > 490:
                db("    Saving WUNDERMODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, text = "WWWWWWWWWWWunderModel"))

            if onlyAcceptBetterModels:
                if averageReturn > highestAverage:
                    db("    Accepting new model")
                    highestAverage = averageReturn
                    agent.saveModel("tempModel")
                else:
                    db("    Rejecting new model")
                    agent.loadModel("tempModel")
            elif onlySaveBestModel:
                if averageReturn > highestAverage:
                    db("    Saving best model")
                    db("      highestAverage " + str(highestAverage))
                    db("      averageReturn " + str(averageReturn))
                    highestAverage = averageReturn
                    agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, text = "bestModel"))
                    # agent.saveModel(saveName + "_eval-" + str(evaluation) + "_bestModel" )
            else:
                db("    Saving model for current batch")
                agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, batch = batch))
                # agent.saveModel(saveName + "_eval-" + str(evaluation) + "_batch-" + str(batch) + "_" + str(batch) )

        agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, text = "finalModel"))
        # agent.saveModel(saveName + "_eval-" + str(evaluation) + "_finalModel")

        averageRunReturns.append(list(averageBatchReturns))
        db("  Saving evaluation results")
        np.save(createSaveName(name = saveName, evaluation=evaluation, text = "finalEvaluationReturn"), averageBatchReturns)
        # np.save(saveName + "_" + "eval-" + str(evaluation) + "_finalReturn", averageBatchReturns)
        # loadTest = np.load(environmentName + "_" + saveName + "_" + str(evaluation) + ".npy")

        # DB()
        # DB("Final Results: [nr of training runs, average score]")
        # DB(loadTest)
        DB("  Saving final model")
        agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, text = "finalModel"))
        # agent.saveModel(saveName + "_eval_" + str(evaluation) + "_finalModel")
    db("Saving final evaluation results")
    np.save(createSaveName(name = saveName, text = "finalResults"), np.array(averageRunReturns))
    loadTest = np.load(createSaveName(name = saveName, text = "finalResults") + ".npy")
    print loadTest

    # for a in averageRunReturns:
    #     print a
if __name__ == "__main__":
    solveEnvironment()
