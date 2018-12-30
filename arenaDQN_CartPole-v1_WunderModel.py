import gym
import numpy as np
from agents.DQNAgent import DQNAgent as currentAgent
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

environmentName = "Acrobot-v1"
saveName = "Acrobot-v1_DQN_2x20"
modelName = "WWWWWWWWWWWunderModel"

if len(sys.argv) > 2:
    saveName += "_" + str(sys.argv[1])
renderEnvironment = True
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

# Run a bunch of measurements of agent performance as a function of nr of training runs
def runModel():

    environment = gym.make(environmentName)

    # Measure performance as a function nr of trainings
    for evaluation in range(nrOfEvaluations):
        DB("Evaluation nr: " + str(evaluation))

        # Change the agent by importing a different agent class
        agent = currentAgent(environment)
        agent.loadModel(modelName)

        # averageBatchReturns = np.zeros((nrOfBatches,2), dtype = int) # score for each batch
        # highestAverage = -1000000

        # Evaluate agent performance
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
                    db("      Evaluation Run: " + str(i) +  " return: " + str(currentReturn), "")
                    returnSum += currentReturn
                    break

        # calucate average return
        averageReturn = returnSum / runsPerEvaluation
        trainCount = (batch + 1) * trainingsPerBatch
        averageBatchReturns[batch] = [trainCount, averageReturn]
        DB("    Average score " + str(averageReturn))

if __name__ == "__main__":
    runModel()
