import numpy
import csv
import os

os.chdir('./data')

Y_train = numpy.loadtxt("y_04.csv", dtype="int", delimiter=",", usecols=0)

t = Y_train.shape[0]

csvFile = open("x_04.csv", "r")
reader = csv.reader(csvFile)
X_train = numpy.zeros((t, 257), dtype=int)
for i in range(t):
    X_train[i][0] = Y_train[i]

fout = open("single byte count-04.txt", "w")
now = 0
maxnumber = 0
for line in reader:
    length = len(line)
    '''for i in range(length):
        if int(line[i]) > maxnumber:
            maxnumber = int(line[i])'''
    '''print(now, length)'''
    for i in range(length):
        t = int(line[i])
        X_train[now][t+1] = X_train[now][t+1] + 1
    fout.write(str(X_train[now][0]))
    for i in range(256):
        fout.write(' ' + str(i) + ':' + str(X_train[now][i+1]))
    fout.write('\n')
    now = now+1
'''print(X_train)'''

'''print(maxnumber)'''

csvFile.close()
fout.close()
