import csv
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage.metrics import structural_similarity as ssim
import os

# do a MDM (molecular distance map) of lungfish, bony fish, cartilage fish, amphibians
# Could try doing a MDM for all of them (let's see how long it takes)

def parse_sequence(filename):
    with open(filename) as inputfile:
        next(inputfile)
        results = list(csv.reader(inputfile))
    seq = ''
    for r in results[:-1]:
        seq = seq + r[0]
    return seq

def dna_cgr(seq):
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
    plt.figure()
    plt.plot(coord[0, :], coord[1, :], 'k,')
    plt.axis('off')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', 'box')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def make_CGR(fasta_file, output_file):
    sequence = parse_sequence(fasta_file)
    coordinate = dna_cgr(sequence)
    cgr = plot_CGR(coordinate, output_file)

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
