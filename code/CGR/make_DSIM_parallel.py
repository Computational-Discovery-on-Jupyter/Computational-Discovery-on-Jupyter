import itertools
import numpy as np
import multiprocessing as mp
import time
import skimage.io
from skimage.metrics import structural_similarity as ssim
import os
import pickle

# Open saved dictionary containing CGR of all vertebrate DNA sequences
file = open("animal_cgr.pkl", "rb")
animalCGR = pickle.load(file)
file.close()

animalCode = list(animalCGR.keys())

# Test multiprocessing on first five animals
# animalCode = animalCode[:5]
# print(animalCode)

# Calculate the dissimilary matrix
n = len(animalCode)
def calc_dsim(animal1, animal2):
    figure1 = animalCGR[animal1]['CGR']
    figure2 = animalCGR[animal2]['CGR']
    return(1 - ssim(figure1, figure2))

# Parallelize the process
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    args = [(i, j) for i, j in list(itertools.combinations(animalCode, 2))]
    # print(args[0])
    # print(calc_dsim(*args[0]))

    print('Finish creating args')
    results = pool.starmap(calc_dsim, args)
    print('Done calculating results')
    pool.close()
    pool.join()

    dsim_mat = np.zeros((n, n))
    for i_tuple, result in zip([(i, j) for i, j in list(itertools.combinations(range(n), 2))], results):
        i, j = i_tuple
        dsim_mat[i, j] = result
        dsim_mat[j, i] = result

    # print(dsim_mat)
    dsim_mat.dump('DSSIM_matrix.dat')
