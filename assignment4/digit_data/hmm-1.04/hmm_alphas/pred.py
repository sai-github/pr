import numpy as np
import glob
import sys

def main(org_class):
	c1 = np.loadtxt("hmm_alphas/1.out")
	c2 = np.loadtxt("hmm_alphas/2.out")
	c3 = np.loadtxt("hmm_alphas/3.out")

	scores_class = np.zeros((c1.shape[0], 2))

	c_1 = 0
	c_2 = 0
	c_3 = 0
	for i in range(c1.shape[0]):
		if c1[i]> max(c2[i], c3[i]):
			scores_class[i] = np.array(['%.6f' %c1[i], '1'])
			c_1+=1
		elif c2[i]> max(c1[i], c3[i]):
			scores_class[i] = np.array(['%.6f' %c2[i], '2'])
			c_2+=1
		else:
			scores_class[i] = np.array(['%.6f' %c3[i], '3'])
			c_3+=1

	with open('hmm_alphas/acc.log', 'a') as f:
		f.write('Acc  (class): '+ str(org_class) + ' ' +str(['%.2f' % (i/c1.shape[0]) for i in [c_1, c_2, c_3]][org_class-1])+'\n') #0,1,2

	np.savetxt("hmm_alphas/scores_class", scores_class)

if __name__ == '__main__':
	main(int(sys.argv[1]))
