import numpy as np
import glob

c1 = np.loadtxt("hmm_alphas/1/1.out")
c2 = np.loadtxt("hmm_alphas/1/2.out")
c3 = np.loadtxt("hmm_alphas/1/3.out")

scores_class = np.zeros((c1.shape[0], 2))

c_1 = 0
c_2 = 0
c_3 = 0
for i in range(c1.shape[0]):
	if c1[i]> max(c2[i], c3[i]):
		scores_class[i] = np.array([c1[i], 1])
		c_1+=1
	elif c2[i]> max(c1[i], c3[i]):
		scores_class[i] = np.array([c2[i], 2])
		c_2+=1
	else:
		scores_class[i] = np.array([c3[i], 3])
		c_3+=1


np.savetxt("hmm_alphas/1/scores_class", scores_class)
