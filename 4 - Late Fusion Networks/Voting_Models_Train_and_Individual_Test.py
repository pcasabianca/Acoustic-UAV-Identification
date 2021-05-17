from datetime import datetime
from termcolor import colored

# Timer.
startTime = datetime.now()

# Training the 10 single CRNN models.

print(colored("\nStarting Training Runs of CRNN Model", "green"))

print(colored("\nCRNN is on run 1 of its training", "green"))
stream_1 = open("CRNN_Trainer_1.py")
read_file_1 = stream_1.read()
exec(read_file_1)

print(colored("\nCRNN is on run 2 of its training", "green"))
stream_2 = open("CRNN_Trainer_2.py")
read_file_2 = stream_2.read()
exec(read_file_2)

print(colored("\nCRNN is on run 3 of its training", "green"))
stream_3 = open("CRNN_Trainer_3.py")
read_file_3 = stream_3.read()
exec(read_file_3)

print(colored("\nCRNN is on run 4 of its training", "green"))
stream_4 = open("CRNN_Trainer_4.py")
read_file_4 = stream_4.read()
exec(read_file_4)

print(colored("\nCRNN is on run 5 of its training", "green"))
stream_5 = open("CRNN_Trainer_5.py")
read_file_5 = stream_5.read()
exec(read_file_5)

print(colored("\nCRNN is on run 6 of its training", "green"))
stream_6 = open("CRNN_Trainer_6.py")
read_file_6 = stream_6.read()
exec(read_file_6)

print(colored("\nCRNN is on run 7 of its training", "green"))
stream_7 = open("CRNN_Trainer_7.py")
read_file_7 = stream_7.read()
exec(read_file_7)

print(colored("\nCRNN is on run 8 of its training", "green"))
stream_8 = open("CRNN_Trainer_8.py")
read_file_8 = stream_8.read()
exec(read_file_8)

print(colored("\nCRNN is on run 9 of its training", "green"))
stream_9 = open("CRNN_Trainer_9.py")
read_file_9 = stream_9.read()
exec(read_file_9)

print(colored("\nCRNN is on run 10 of its training", "green"))
stream_10 = open("CRNN_Trainer_10.py")
read_file_10 = stream_10.read()
exec(read_file_10)


# Getting voted predictions from each model.
print(colored("\nModel 1 is being assessed", "green"))
stream_1 = open("Perf_Hard_and_Soft_Voting_Prep_1.py")
read_file_1 = stream_1.read()
exec(read_file_1)

print(colored("\nModel 2 is being assessed", "green"))
stream_2 = open("Perf_Hard_and_Soft_Voting_Prep_2.py")
read_file_2 = stream_2.read()
exec(read_file_2)

print(colored("\nModel 3 is being assessed", "green"))
stream_3 = open("Perf_Hard_and_Soft_Voting_Prep_3.py")
read_file_3 = stream_3.read()
exec(read_file_3)

print(colored("\nModel 4 is being assessed", "green"))
stream_4 = open("Perf_Hard_and_Soft_Voting_Prep_4.py")
read_file_4 = stream_4.read()
exec(read_file_4)

print(colored("\nModel 5 is being assessed", "green"))
stream_5 = open("Perf_Hard_and_Soft_Voting_Prep_5.py")
read_file_5 = stream_5.read()
exec(read_file_5)

print(colored("\nModel 6 is being assessed", "green"))
stream_6 = open("Perf_Hard_and_Soft_Voting_Prep_6.py")
read_file_6 = stream_6.read()
exec(read_file_6)

print(colored("\nModel 7 is being assessed", "green"))
stream_7 = open("Perf_Hard_and_Soft_Voting_Prep_7.py")
read_file_7 = stream_7.read()
exec(read_file_7)

print(colored("\nModel 8 is being assessed", "green"))
stream_8 = open("Perf_Hard_and_Soft_Voting_Prep_8.py")
read_file_8 = stream_8.read()
exec(read_file_8)

print(colored("\nModel 9 is being assessed", "green"))
stream_9 = open("Perf_Hard_and_Soft_Voting_Prep_9.py")
read_file_9 = stream_9.read()
exec(read_file_9)

print(colored("\nModel 10 is being assessed", "green"))
stream_10 = open("Perf_Hard_and_Soft_Voting_Prep_10.py")
read_file_10 = stream_10.read()
exec(read_file_10)

# Timer output.
time = datetime.now() - startTime
print(time)
