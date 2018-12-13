


import csv
import numpy as np


# Read CSV file
with open('test_30.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

# Parse Data to np array
myFile = np.genfromtxt('test_30.csv', delimiter=',')
data_example = myFile[1:, 2:]

# One-Hot encoding for targets (of three classes)
target_example = np.zeros([len(data)-1, 3])
for i in range(1, len(data)):
    if data[i][1] == 'c1':
        target_example[i-1, 0] = 1
    elif data[i][1] == 'c2':
        target_example[i-1, 1] = 1
    elif data[i][1] == 'c3':
        target_example[i-1, 2] = 1

print(np.shape(data_example))
print(np.shape(target_example))
