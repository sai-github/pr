import numpy as np
import glob

c1 = np.loadtxt('1/actuallabel_1_model_1')
c2 = np.loadtxt('1/actuallabel_1_model_2')
c3 = np.loadtxt('1/actuallabel_1_model_3')

scores_class = np.zeros((c1.shape[0], 2))
for i in range(c1.shape[0]):
	if c1[i]> max(c2[i], c3[i]):
		scores_class[i] = np.array([c1[i], 1])
	elif c2[i]> max(c1[i], c3[i]):
		scores_class[i] = np.array([c2[i], 2])
	else:
		scores_class[i] = np.array([c3[i], 3])

np.savetxt('1/scores_class', scores_class)
