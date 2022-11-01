import numpy as np


def parce_params(file_name: str) -> np.array:

    with open(file_name, 'r') as file:
        line = file.readline()
        lines = line.split(',{')

    for i in range(len(lines)):

        line = lines[i]
        line = line.split(',')
        line = line[1].split()
        lines[i] = float(line[1])

    return np.array(lines)
