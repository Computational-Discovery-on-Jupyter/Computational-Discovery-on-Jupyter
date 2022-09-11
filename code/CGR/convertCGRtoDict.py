import skimage.io
from skimage.metrics import structural_similarity as ssim
import os
import pickle

# Read all CGR figures into matrices and save as dictionary
animalCGR = {}
for (dirpath, dirnames, filenames) in os.walk('CGR'):
    for name in filenames:
        animal = name[:-4]
        temp = {}
        a_class = dirpath[4:]
        temp['class'] = a_class
        figure = skimage.io.imread(os.path.join(dirpath, name), as_gray=True)
        temp['CGR'] = figure
        animalCGR[animal] = temp
# print(animalCGR)
file = open('animal_cgr.pkl', 'wb')
pickle.dump(animalCGR, file)
file.close()
