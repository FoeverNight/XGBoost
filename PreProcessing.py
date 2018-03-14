import numpy
import csv
import os

os.chdir('./data')

Y_train = numpy.loadtxt("y_05.csv", dtype="int", delimiter=",", usecols=0)

t = Y_train.shape[0]

csvFile = open("x_05.csv", "r")
reader = csv.reader(csvFile)
X_train = numpy.zeros((t, 257), dtype=int)
for i in range(t):
    X_train[i][0] = Y_train[i]

now = 0
maxnumber = 0
for line in reader:
    length = len(line)
    '''for i in range(length):
        if int(line[i]) > maxnumber:
            maxnumber = int(line[i])'''
    print(now, length)
    for i in range(length):
        t = int(line[i])
        X_train[now][t+1] = X_train[now][t+1] + 1
    now = now+1
'''print(X_train)'''

'''print(maxnumber)'''

numpy.savetxt('single byte count-05.csv', X_train, fmt="%d", delimiter=',')

csvFile.close()
