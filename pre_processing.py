import csv
import numpy as np

def load_data(filename):
    with open(filename,'r') as file:
        reader = csv.reader(file)
        columnNames = next(reader)
        rows = np.array(list(reader), dtype=float)
        return  columnNames,rows


def separate(columnNames, rows, index):
    labelColumnIndex = columnNames.index(index)
    ys = rows[:, labelColumnIndex]
    xs = np.delete(rows,labelColumnIndex,axis=1)
    del columnNames[labelColumnIndex]
    return columnNames, xs, ys






