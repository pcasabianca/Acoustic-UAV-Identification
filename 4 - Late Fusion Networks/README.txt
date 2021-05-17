After all voting models are trained, "Perf_Hard_and_Soft_Voting_Prep.py" is run for each model to get voted predictions
from each model. These can be saved in the form "voted_n.json", where n is the model number.

This can also be setup and run by "Voting_Models_Train_and_Individual_Test.py".

All the model accuracies from the Validation 2 subset test must be input into one file, as is done with
"all_model_acc.json", before the final scripts can be run. This allows weights for the weighted soft voting model to be
calculated.

"Performance_Soft_Voting_Calcs.py" and "Performance_Hard_Voting_Calcs.py" can then be run.