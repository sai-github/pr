import numpy as np
import glob

files = glob.glob('*_')

file_lines = []
for file in files:
    lines = None
    with open(file) as f:
        lines = f.readlines()
    file_lines.append(lines)

file_lines = np.array(file_lines)

file_count = 0
for file in file_lines:
    filename = 'chmms_m/'+files[file_count]+'.hmm'
    file_count+=1
    nb_state = int(file[0].strip().split(' ')[1])//3

    for i in range(1,3):
        state_end_index = 2+3*nb_state*i-1
        l_temp = file[state_end_index-1].strip().split(' ') #19
        l_temp[0] = '%.6f' %0.8
        file[state_end_index-1] = ''.join(l_temp)+' \n'

        r_temp = file[state_end_index].strip().split(' ') #20
        r_temp[0] = '%.6f' %0.2
        file[state_end_index] = ''.join(r_temp)+' \n'

    line_val = 0
    with open(filename, 'w') as f:
        n_max = nb_state*3*3 +2 #3-rows in text file, 3-combinations
        while(line_val<n_max):
            f.write(file[line_val])
            line_val+=1
