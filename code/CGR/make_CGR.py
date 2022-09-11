import csv
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage.metrics import structural_similarity as ssim
import os

def parse_sequence(filename):
    """
    Reads fasta file and convert it to string that Python can use
    Input:
    filename: (string) path to fasta file
    """
    with open(filename) as inputfile:
        next(inputfile)
        results = list(csv.reader(inputfile))
    seq = ''
    for r in results[:-1]:
        seq = seq + r[0]
    return seq

def dna_cgr(seq):
    """
    Create array with the coordinate of all the points that corresponds with the DNA sequence
    Input:
    seq: (string) DNA sequence
    """
    N = len(seq)
    dataPoints = np.zeros((2, N+1))
    dataPoints[:, 0] = np.array([0.5, 0.5])
    for i in range(1, N+1):
        if seq[i-1] == 'A':
            corner = np.array([0, 0])
        elif seq[i-1] == 'C':
            corner = np.array([0, 1])
        elif seq[i-1] == 'G':
            corner = np.array([1, 1])
        else:
            corner = np.array([1, 0])
        dataPoints[:, i] = 0.5*(dataPoints[:, i-1] + corner)
    return(dataPoints)

def plot_CGR(coord, filename):
    """
    Plot CGR
    Inputs:
    coord: (numpy.array) array with coordinates corresponding to CGR of DNA sequence
    filename: (string) path to directory to save figure
    """
    plt.figure()
    plt.plot(coord[0, :], coord[1, :], 'k,')
    plt.axis('off')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def make_CGR(fasta_file, output_file):
    """
    Function that calls all previous functions to make CGR for DNA sequence
    Inputs:
    fasta_file: (string) path to fasta file
    output_file: (string) path to directory to save CGR figure
    """
    sequence = parse_sequence(fasta_file)
    coordinate = dna_cgr(sequence)
    cgr = plot_CGR(coordinate, output_file)

# Go through all fasta files and create CGR
f = []
for (dirpath, dirnames, filenames) in os.walk('FASTA files'):
    for name in filenames:
        f.append(os.path.join(dirpath, name))

for file in f:
    index = file.find('/', 12)
    animal_class = file[12:index]
    animal_id = file[index+1:-6]
    path = 'CGR/'+animal_class
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    make_CGR(file, path+'/'+animal_id+'.png')
