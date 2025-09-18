import numpy as np
from scipy.stats import special_ortho_group


def generate_random_rotations (N: int, M: int = 1) -> np.ndarray:

	'''
	Generates M random matrices in SO(N)
	'''

	R = special_ortho_group.rvs(N, size=M)

	return R


if __name__ == '__main__':

	N = int(input("N = "))
	M = int(input("M = "))

	Rs = generate_random_rotations (N, M)

