from google.colab import drive

gdrive_root = '/content/gdrive'

drive.mount(gdrive_root)

gdrive_path = f'{gdrive_root}/My Drive/GBDT-Cartpole'

import gym
import numpy as np
currentAgent = GBDTAgent
import sys

################################################################################################################
#
# Run directions:
#     - python arena.py <add this text to save files> <dir for save results>
#     - fx: python arena.py runNr2 measurements
#     - Arguments can be omitted
#
################################################################################################################
#
# Main purpose:
#     Train an evaluate performance of an agent playing a game in an OpenAI Gym environment.
#
################################################################################################################
#
# Detailed description
#     - Run nrOfEvaluations runs starting form a fresh agent every evaluation (random weights, empty tables, etc).
#     - Each evaluation alternates between training batches and evaluations.
#     - Every evaluation the average return is calculated after every batch using runsPerEvaluation simulations.
#     - Saves the results to file : "<saveName>_eval-<evaluation>_<x>" where <x>:
#       - "bestModel.pth.tar" : The best model for the corresponding evaluation
#       - "finalEvaluationReturn.npy" : Numpy array of shape: (nrOfBatches*trainingsPerBatch, 2)
#                                       Each line containing [training count, average return]
#       - "finalModel" : Final model for each evaluation
#       - "WWWWWWWWWWWunderModel.pth.tar" : A model with super high return
#
################################################################################################################
#
# Structure:
#    for nrOfEvaluations:
#        for nrOfBatches:
#            for trainingsPerBatch:
#            for runsPerEvaluation:
#
################################################################################################################
#
# Settings
#

environmentName = "CartPole-v1"
saveName = "CartPole_GBDTA_100"
if len(sys.argv) > 2:
    saveName += "_" + str(sys.argv[1])
renderEnvironment = False
onlyAcceptBetterModels = True
onlySaveBestModel = True
saveResultsToArgDir = True

wunderModelLimit = 500

nrOfEvaluations = 10
nrOfBatches = 50
trainingsPerBatch = 5
runsPerEvaluation = 100

# Verbose Level 1
verbose1 = True

# Verbose Level 2
verbose2 = True

# Verbose Level 1
def db(head = "", tail = ""):
    if verbose1:
        print(str(head) + str(tail))
        #print str(head) + str(tail)

# Verbose Level 2
def DB(head = "", tail=""):
    if verbose2:
        print(str(head) + str(tail))
        #print str(head) + str(tail)

# Construct save name for results
def createSaveName(name = False, evaluation = -1, batch = -1, text = None):
    tempText = gdrive_path + '/'
    if name:
        tempText += saveName + "_"
    if evaluation != -1:
        tempText += "eval-" + str(evaluation) + "_"
    if batch != -1:
        temptext += "batch-" + str(batch) + "_"
    if text != None:
        tempText += text
    return tempText

# Run a bunch of measurements of agent performance as a function of nr of training runs
def runMeasurements():

    averageRunReturns = []
    environment = gym.make(environmentName)

    # Measure performance as a function nr of trainings
    for evaluation in range(nrOfEvaluations):
        DB("Evaluation nr: " + str(evaluation))

        # Change the agent by importing a different agent class
        agent = currentAgent(environment)

        averageBatchReturns = np.zeros((nrOfBatches,2), dtype = int) # score for each batch
        highestAverage = -1000000

        # Run batches of alternating trainings and evaluations
        for batch in range(nrOfBatches): # go through all train/evaluation batches
            DB("  Batch nr: ", str(batch))
            DB("    Exporation Rate: " + str(agent.explorationRate))
            DB("    Training...")

            # Train
            for i in range(trainingsPerBatch):
                currentReturn = 0
                currentState = environment.reset()
                while True:
                    #if renderEnvironment: environment.render()
                    action = agent.getNextAction(currentState)
                    successorState, reward, done, info = environment.step(action)
                    currentReturn += reward
                    agent.mem(currentState, action, reward, done, successorState)
                    if done:
                        # reward = 0
                        db("      Training Run: " + str(i) + ", exploration rate: " + str(agent.explorationRate) + " return: " + str(currentReturn), "")
                        agent.updatePolicy()
                        break
#                     agent.updatePolicy(currentState, action, reward, done, successorState)
                    currentState = successorState

            DB("    Evaluating agent...")
            returnSum = 0

            # Evaluate agent performance
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
                        db("      Evaluation Run: " + str(i) + ", exploration rate: " + str(agent.explorationRate) + " return: " + str(currentReturn), "")
                        returnSum += currentReturn
                        break

            # calucate average return
            averageReturn = returnSum / runsPerEvaluation
            trainCount = (batch + 1) * trainingsPerBatch
            averageBatchReturns[batch] = [trainCount, averageReturn]
            DB("    Average score over " + str(runsPerEvaluation) + " evaluation runs after " + str(trainCount) + " trainings = " + str(averageReturn) , "")

            # Saving super high performance models
            if averageReturn > int(wunderModelLimit*0.9):
                DB("    Saving WUNDERMODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, text = "WWWWWWWWWWWunderModel"))

            # Save Models
            if onlyAcceptBetterModels:
                if averageReturn > highestAverage:
                    DB("    Accepting new model")
                    highestAverage = averageReturn
                    agent.saveModel(f'{gdrive_path}/{saveName}' + "_tempModel")
                else:
                    DB("    Rejecting new model")
                    agent.loadModel(f'{gdrive_path}/{saveName}' + "_tempModel")
            elif onlySaveBestModel:
                if averageReturn > highestAverage:
                    DB("    Saving best model")
                    highestAverage = averageReturn
                    agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, text = "bestModel"))
            else:
                DB("    Saving model for current batch")
                agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, batch = batch))

        # Save final model for the current evaluation
        agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, text = "finalModel"))
        averageRunReturns.append(list(averageBatchReturns))

        DB("  Saving evaluation results")
        np.save(createSaveName(name = saveName, evaluation=evaluation, text = "finalEvationReturn"), averageBatchReturns)

        DB("  Saving final model")
        agent.saveModel(createSaveName(name = saveName, evaluation=evaluation, text = "finalModel"))

    DB("Saving final evaluation results")
    np.save(createSaveName(name = saveName, text = "finalResults"), np.array(averageRunReturns))

    # loadTest = np.load(createSaveName(name = saveName, text = "finalResults") + ".npy")

if __name__ == "__main__":
     runMeasurements()
