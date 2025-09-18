import numpy as np
from scipy.stats import special_ortho_group


def generate_random_rotations (N: int, M: int = 1) -> np.ndarray:

	'''
	Generates M random matrices in SO(N)
	'''

	R = special_ortho_group.rvs(N, size=M)

	return R


def augment_data (X: np.ndarray, M: int, flatten: bool = True) -> np.ndarray:
	'''
	Create copies of each point in the input data array.

	Arguments
	---------

	X: (..., D) np.ndarray
		Input data

	M: int
		Number of randomly rotated copies for each point

	Returns
	-------

	Y: (..., M, D)
		Augmented data
	'''

	D = X.shape[-1]

	Rs = generate_random_rotations(D, M)
	Y = np.einsum('...i,jki->j...k', X, Rs)

	if flatten:
		Y = np.concatenate(Y, axis=0)

	return Y


if __name__ == '__main__':

	N = tuple([int(n) for n in input("N = ").split(',')])
	M = int(input("M = "))
	D = int(input("D = "))

	X = np.random.randn(*N, D)

	Y = augment_data(X, M)

	print("X.shape =", X.shape)
	print("Y.shape =", Y.shape)
