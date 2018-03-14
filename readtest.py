import csv

csvfile = open("y_04.csv", "r")
reader = csv.reader(csvfile)
y = list(range(120000))
i = 0

for line in reader:
    i = i+1
    y[i] = int(line[0])*10 + int(line[1])
    print(y[i])

csvfile.close()
