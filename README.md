# RL-Alcomp
## Final project in Reinforcement Learning @ Reykjavik University

### Main purpose:
Train and evaluate performance of an agent playing a game in an OpenAI Gym environment.

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
