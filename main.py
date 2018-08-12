import numpy as np
import cv2

from random import randint, shuffle
from skimage.measure import compare_ssim as ssim
from operator import itemgetter

"""
TODO:
    - put Rectangle class back
    - change mutation to randomly pick and choose a field or fields of a rectangle to change
    - use collections.namedtuple for points and colour
    x change midpoint between child generation
    x randomise order of parents before generating child.

"""

class Rectangle:
    def __init__(self):
        self.point_one = (randint(0, 513), randint(0, 513))
        self.point_two = (randint(0, 513), randint(0, 513))
        self.colour = (randint(0,255), randint(0, 255), randint(0, 255),)
        self.alpha = randint(0,255)

    

class Organism:
    def __init__(self, chromosome_length):
        self.rectangles = []

        for _ in range(chromosome_length):
            self.rectangles.append(Rectangle())

        self.chromosome = ""

        for rectangle in self.rectangles:
            self.chromosome += format(rectangle.point_one[0],'b').zfill(16)
            self.chromosome += format(rectangle.point_one[1],'b').zfill(16)
            self.chromosome += format(rectangle.point_two[0],'b').zfill(16)
            self.chromosome += format(rectangle.point_two[1],'b').zfill(16)
            self.chromosome += format(rectangle.colour[0],'b').zfill(16)
            self.chromosome += format(rectangle.colour[1],'b').zfill(16)
            self.chromosome += format(rectangle.colour[2],'b').zfill(16)
            self.chromosome += format(rectangle.alpha,'b').zfill(16)

def mutate(chromosome):
    while randint(1, mutation_chance) == 1:
        start = randint(0, len(chromosome)-200)
        stop = randint(start+200, len(chromosome))
        chromosome = chromosome[:start] + ''.join([str((int(char)-1)**2) for char in chromosome[start:stop]]) + chromosome[stop:]
    return chromosome

def evolve(ssims):
    survivors = [org[0] for org in sorted(ssims, key=itemgetter(1))[int(len(ssims)/2):]]

    new_population = []
    for survivor in survivors:
        new = Organism(genome_size)
        new.chromosome = survivor
        new_population.append(new)

    shuffle(survivors)
    for i in range(0, len(survivors), 2):
        midpoint = randint(0, len(survivors[i]))
        new = Organism(genome_size)
        new.chromosome = mutate(survivors[i][midpoint:] + survivors[i+1][:midpoint])
        new_population.append(new)

        midpoint = randint(0, len(survivors[i]))
        new = Organism(genome_size)
        new.chromosome = mutate(survivors[i+1][midpoint:] + survivors[i][:midpoint])
        new_population.append(new)

    return new_population

def assemble_from_chromosome(chromosome):
    # break chromosome into separate rectangles
    rectangles = []
    for gene in [chromosome[i:i+16*8] for i in range(0, len(chromosome), 16*8)]:
        # break rectangle into constituent 7 parts (x1, y1. x2, y2, r, g, b, a)
        attributes = [int(gene[i:i+16],2) for i in range(0, len(gene), 16)]
        # format attributes into (x1, y1), (x2, y2), (r, g, b), a
        attributes, alpha = [(attributes[0], attributes[1]), (attributes[2], attributes[3]), (attributes[4], attributes[5], attributes[6])], attributes[7]/255
        rectangles.append((attributes, alpha,))

    return rectangles

def create_image(rectangles, img):
    for rectangle in rectangles:
        this = np.copy(img)
        cv2.rectangle(this, *rectangle[0], -1)
        cv2.addWeighted(this, rectangle[1], img, 1-rectangle[1], 0, img)

    return img

def init(population_size, genome_size):
    population = []
    for _ in range(population_size):
        population.append(Organism(genome_size))

    return population

def main(population):
    for gen in range(1000):
        ssims = []
        for org in population:
            img = np.copy(blank)
            rectangles = assemble_from_chromosome(org.chromosome)
            org.image = create_image(rectangles, img)
            img_ssim = (ssim(source, org.image, multichannel=True)/2)+0.5
            ssims.append((org.chromosome, img_ssim))

        ssims = sorted(ssims, key=itemgetter(1))[::-1]
        print(f"Gen {gen}: {round(ssims[0][1], 5)} (diff: {round(ssims[0][1]-ssims[-1][1], 5)})")
        population = evolve(ssims)

    return ssims

source = cv2.resize(cv2.imread('input.jpg'), (0,0), fx=0.25, fy=0.25)
blank = np.zeros((source.shape[0],source.shape[1],3), np.uint8)
blank[:] = (255,255,255)
mutation_chance = 2
genome_size = 40
population_size =  40
population = init(population_size, genome_size)
ssims = main(population)

#cv2.imshow('',assemble_from_chromosome(ssims[0][0], np.copy(blank)))
#cv2.waitKey(0)