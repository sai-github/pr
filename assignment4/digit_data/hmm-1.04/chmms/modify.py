import numpy as np
import glob

files = glob.glob('*')

for file in files:
    lines = []
    with open(file) as f:
        lines.append(f.readlines())
    nb_state = int(lines[0].strip().split(' ')[1])
    nb_symbols = int(lines[1].strip().split(' ')[1])

    for data in lines[3:]:

    
