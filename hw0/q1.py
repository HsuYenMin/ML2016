import numpy as np
import sys
def getValue(item):
    return float(item)
table=[]
filename = sys.argv[2]
index = int(sys.argv[1])
with open(filename,"r") as f:
    for line in f:
        table.append([s for s in line.split()])
column=[]
for line in table:
    column.append(line[index])
column = sorted(column, key = getValue)
with open("ans1.txt",'w') as f:
    str1 = ','.join(str(e) for e in column)
    f.write(str1)
        



