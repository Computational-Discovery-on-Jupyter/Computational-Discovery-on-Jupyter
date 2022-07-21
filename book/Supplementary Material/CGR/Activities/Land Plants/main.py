import math
import numpy as np
import random
import csv
import skimage.io
from skimage.metrics import structural_similarity as ssim
import pickle
import pandas as pd
import multiprocessing as mp
import os
import itertools
import matplotlib.pyplot as plt

def n_gon(n, *start):
    if start:
        deg = np.linspace(start[0], start[0] + 360, n+1)
    else:
        deg = np.linspace(0, 360, n+1)
    deg = deg[:-1]
    rad = []
    for d in deg:
        rad.append((d/360)*2*math.pi)
    cor = np.zeros((2, n+1))
    for r in range(len(rad)):
        x = math.cos(rad[r])
        y = math.sin(rad[r])
        cor[:, r] = np.array([x, y])
    cor[:, n] = cor[:, 0]
    return cor

def dividingRateAlmeida(n):
    k = round((n+2.)/4.)
    s_num = 2*math.cos(math.pi*((1/2) - (k/n))) - 2*math.cos(math.pi*((1/2)-(1/(2*n))))*math.cos((2*k-1)*(math.pi/(2*n)))*(1 + (math.tan((2*k-1)*(math.pi/(2*n))))/(math.tan(math.pi-((n+2*k-2)*(math.pi/(2*n))))))
    s_den = 2*math.cos(math.pi*((1/2) - (k/n)))
    return s_num/s_den

def parse_sequence(filename):
    with open(filename) as inputfile:
        next(inputfile)
        results = list(csv.reader(inputfile))
    seq = ''
    for r in results[:-1]:
        seq = seq + r[0]
    return seq

def protein_cgr(seq):
    N = len(seq)
    AA = 'ARNDCQEGHILKMFPSTWYVBZ'
    # B = Aspartic acid (D) or Asparagine (N)
    # Z = Glutamic acid (E) or Glutamine (Q)
    vertices = n_gon(20, 90)
    r = dividingRateAlmeida(20)
    dataPoints = np.zeros((2, N+1))
    for i in range(1, N+1):
        index = AA.index(seq[i-1])
        if index == 20:
            r = random.randint(0, 1)
            if r == 0:
                index = AA.index('D')
            else:
                index = AA.index('N')
        elif index == 21:
            r = random.randint(0, 1)
            if r == 0:
                index = AA.index('E')
            else:
                index = AA.index('Q')
        dataPoints[:, i] = dataPoints[:, i-1] + (vertices[:, index] - dataPoints[:, i-1])*r
    return(vertices, dataPoints)

def plot_CGR(coord, filename):
    plt.figure()
    plt.plot(coord[0, :], coord[1, :], 'k,')
    plt.axis('off')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def make_CGR(fasta_file, output_file):
    sequence = parse_sequence(fasta_file)
    _, coordinate = protein_cgr(sequence)
    cgr = plot_CGR(coordinate, output_file)

# Create CGR of Land Plants
f = []
plant_class_list = []
for (dirpath, dirnames, filenames) in os.walk('Fasta Files'):
    for name in filenames:
        path_str = os.path.join(dirpath, name).replace('\\', '/')
        f.append(path_str)

for file in f:
    index_end = file.rfind('/')
    index_beg = file.rfind('/', 0, index_end-1)
    plant_class = file[index_beg+1:index_end]
    plant_class_list.append(plant_class)
    plant_id = file[index_end+1:-6]
    path = 'CGR/'+plant_class
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    make_CGR(file, path+'/'+plant_id+'.png')

# save plant class list as external file (this will help with colouring the points when plotting)
file = open('plant_class.pkl', 'wb')
pickle.dump(plant_class_list, file)
file.close()

# Create dictionary of CGR of plants
plantCGR = {}
for (dirpath, dirnames, filenames) in os.walk('CGR'):
    for name in filenames:
        plant = name[:-4]
        temp = {}
        index = dirpath.rfind('/')
        p_class = dirpath[index+1:]
        temp['class'] = p_class
        figure = skimage.io.imread(os.path.join(dirpath, name), as_gray=True)
        temp['CGR'] = figure
        plantCGR[plant] = temp

# Create dissimilarity matrix (this process uses multithreading or else it will take too long to execute)
plantCode = list(plantCGR.keys())

n = len(plantCode)
def calc_dssim(plant1, plant2):
    figure1 = plantCGR[plant1]['CGR']
    figure2 = plantCGR[plant2]['CGR']
    return(1 - ssim(figure1, figure2))

if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    args = [(i, j) for i, j in list(itertools.combinations(plantCode, 2))]

    results = pool.starmap(calc_dssim, args)
    pool.close()
    pool.join()

    dssim_plant = np.zeros((n, n))
    for i_tuple, result in zip([(i, j) for i, j in list(itertools.combinations(range(n), 2))], results):
        i, j = i_tuple
        dssim_plant[i, j] = result
        dssim_plant[j, i] = result

    dssim_plant.dump('DSSIM_plants.dat')
