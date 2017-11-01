import numpy as np
import glob
import sys

def main():
	alphas_connected = glob.glob('alphas_connected/*.alpha')
	alpha_mat = []
	for file_path in alphas_connected:
		alpha_mat.append(np.loadtxt(file_path))

	alpha_mat = np.array(alpha_mat).reshape(-1,len(alphas_connected))
	for i in np.argmax(alpha_mat, axis=1):
		print(alphas_connected[i].split('/')[1].split('.')[0])

if __name__ == '__main__':
	main()
