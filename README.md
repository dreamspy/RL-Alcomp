# RL-Alcomp
## Final project in Reinforcement Learning @ Reykjavik University

### Main purpose:
Train and evaluate performance of an agent playing a game in an OpenAI Gym environment.

### Agents
Three agents were implemented:
* Deep Q Learning (DQN) agent, as first introduced in the [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) article by the DeepMind guys.
* Generic Q Learning agent that can be used with any multi-output regression model as an approximator for the state-action value function.
* Tabular Q Learning agent. There is a fully working tabular Q agent in the folder 'tabular'. We did not provide any
results for this agent, since we didn't find it as interesting as the function approximation versions of Q-learning.

### Results:
[Click here for a jupyter notebook with main results from DQN agent and 3 versions of the generic agent](https://github.com/dreamspy/RL-Alcomp/blob/master/Results%20-%20DQN%20agent%20solving%20CartPole-v1%20.ipynb)

### Extra:
See the Tabular agent

### Run directions:

* Command structure: `python arena.py <add this text to save files> <dir for save results>`
* Example: `python arena.py runNr2 measurements`
* Note, arguments can be omitted

### Detailed description

* Run `nrOfEvaluations` runs starting form a fresh agent every evaluation (random weights, empty tables, etc)
	* Each evaluation alternates between training batches and evaluations
	* Every evaluation the average return is calculated after every batch using `runsPerEvaluation` simulations
	* Saves the results to file : `<saveName>_eval-<evaluation>_<x>"` where `<x>` :
		* `bestModel.pth.tar` : The best model for the corresponding evaluation
		* `finalEvaluationReturn.npy` :
		 * Numpy array of shape: `(nrOfBatches*trainingsPerBatch, 2)`
          * Each line containing `[training count, average return]`
		* `finalModel` : Final model for each evaluation
       * `WWWWWWWWWWWunderModel.pth.tar` :  Model with super high return

### Structure:
```
   for nrOfEvaluations:
       for nrOfBatches:
           for trainingsPerBatch:
           for runsPerEvaluation:
```
