import numpy as np
import torch
from scipy.stats import special_ortho_group
from typing import Union

def generate_random_rotations (N: int, M: int = 1) -> np.ndarray:

	'''
	Generates M random matrices in SO(N)
	'''

	R = special_ortho_group.rvs(N, size=M)

	return R


def augment_data (X: Union[np.ndarray,torch.Tensor], M: int, flatten: bool = True) -> Union[np.ndarray, torch.Tensor]:
	'''
	Create copies of each point in the input data array.

	Arguments
	---------

	X: (..., D) np.ndarray or torch.Tensor
		Input data

	M: int
		Number of randomly rotated copies for each point

	Returns
	-------

	Y: (..., M, D) np.ndarray or torch.Tensor
		Augmented data
	'''

	if not len(X):
		return X

	D = X.shape[-1]
	Rs = generate_random_rotations(D, M)

	if isinstance(X, np.ndarray):
		Y = np.einsum('...i,jki->j...k', X, Rs)
		if flatten:
			Y = np.concatenate(Y, axis=0)

	elif isinstance(X, torch.Tensor):
		Rs = torch.tensor(Rs, dtype=torch.float)
		Y = torch.einsum('...i,jki->j...k', X, Rs)
		if flatten:
			Y = np.cat(Y, dim=0)

	return Y


if __name__ == '__main__':

	N = tuple([int(n) for n in input("N = ").split(',')])
	M = int(input("M = "))
	D = int(input("D = "))

	X = np.random.randn(*N, D)

	Y = augment_data(X, M)

	print("X.shape =", X.shape)
	print("Y.shape =", Y.shape)
