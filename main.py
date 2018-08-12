import numpy as np
import cv2

from random import randint
from skimage.measure import compare_ssim as ssim
from operator import itemgetter

class Organism:
    def __init__(self, chromosome_length):

        self.chromosome = ""

        for _ in chromosome_length:
            self.chromosome += format(randint(0, 512),'b').zfill(16)
            self.chromosome += format(randint(0, 512),'b').zfill(16)
            self.chromosome += format(randint(0, 512),'b').zfill(16)
            self.chromosome += format(randint(0, 512),'b').zfill(16)
            self.chromosome += format(randint(0, 255),'b').zfill(16)
            self.chromosome += format(randint(0, 255),'b').zfill(16)
            self.chromosome += format(randint(0, 255),'b').zfill(16)
            self.chromosome += format(randint(0, 255),'b').zfill(16)


def evolve(population):
    population = [org[0] for org in population]
    new_population = []
    survivors = population[int(len(population)/2):]
    for i in range(0, len(survivors), 2):
        midpoint = randint(0, len(survivors[i][0]))


        new = Organism(20)
        new.chromosome = survivors[i]
        if randint(1, mutation_chance) == 1:
            start = randint(0, len(new.chromosome))
            stop = randint(start, len(new.chromosome))
            new.chromosome = new.chromosome[:start] + ''.join([str((int(char)-1)**2) for char in new.chromosome[start:stop]]) + new.chromosome[stop:]
        new_population.append(new)

        new = Organism(20)
        new.chromosome = survivors[i+1]
        if randint(1, mutation_chance) == 1:
            start = randint(0, len(new.chromosome))
            stop = randint(start, len(new.chromosome))
            new.chromosome = new.chromosome[:start] + ''.join([str((int(char)-1)**2) for char in new.chromosome[start:stop]]) + new.chromosome[stop:]
        new_population.append(new)

        new = Organism(20)
        new.chromosome = survivors[i][midpoint:] + survivors[i+1][:midpoint]
        if randint(1, mutation_chance) == 1:
            start = randint(0, len(new.chromosome))
            stop = randint(start, len(new.chromosome))
            new.chromosome = new.chromosome[:start] + ''.join([str((int(char)-1)**2) for char in new.chromosome[start:stop]]) + new.chromosome[stop:]
        new_population.append(new)
        
        new = Organism(20)
        new.chromosome = survivors[i+1][midpoint:] + survivors[i][:midpoint]
        if randint(1, mutation_chance) == 1:
            start = randint(0, len(new.chromosome))
            stop = randint(start, len(new.chromosome))
            new.chromosome = new.chromosome[:start] + ''.join([str((int(char)-1)**2) for char in new.chromosome[start:stop]]) + new.chromosome[stop:]
        new_population.append(new)

    return new_population 

def main(population):
    for gen in range(1000):
        ssims = []
        for org in population:
            img = np.copy(blank)
            # break chromosome into separate rectangles
            for chromosome in [org.chromosome[i:i+16*8] for i in range(0, len(org.chromosome), 16*8)]:
                # break rectangle into constituent 7 parts (x1, y1. x2, y2, r, g, b, a)
                attributes = [int(chromosome[i:i+16],2) for i in range(0, len(chromosome), 16)]
                # format attributes into (x1, y1), (x2, y2), (r, g, b), a
                attributes, alpha = [(attributes[0], attributes[1]), (attributes[2], attributes[3]), (attributes[4], attributes[5], attributes[6])], attributes[7]/255
                this = np.copy(img)
                cv2.rectangle(this, *attributes, -1)
                cv2.addWeighted(this, alpha, img, 1-alpha, 0, img) 

            org.image = img
            #cv2.imshow('', img)
            #cv2.waitKey(0)
            ssims.append((org.chromosome, (ssim(source, img, multichannel=True)/2)+0.5,))

        ssims = sorted(ssims, key=itemgetter(1))[::-1]
        print(ssims[0][1])    
        population = evolve(ssims)


source = cv2.resize(cv2.imread('input.jpg'), (0,0), fx=0.25, fy=0.25)

blank = np.zeros((source.shape[0],source.shape[1],3), np.uint8)
blank[:] = (255,255,255)

population = []
population_size = 40
mutation_chance = 8
for _ in range(population_size):
    population.append(Organism(40))

main(population)
