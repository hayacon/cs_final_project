import random
import copy
import numpy as np


class GA_ML:
    def __init__(self):
        pass

    @staticmethod
    def get_random_layer():
        gene = [0, 0, 0]
        gene[0] = random.randint(0, 4)
        if gene[0] == 0:
            gene[1] = random.randint(1, 10)
            gene[2] = random.randint(1, 300)
        elif gene[0] == 1:
            gene[2] = 0
            gene[1] = random.randint(2, 300)
        elif gene[0] == 2:
            gene[2] = 0
            gene[1] = random.randint(2, 300)
        elif gene[0] == 3:
            gene[2] = 0
            gene[1] = random.randint(2, 300)
        elif gene[0] == 4:
            gene[2] = 0
            gene[1] = round(random.uniform(0, 1), 1)

        return gene

    @staticmethod
    def crossover(g1, g2):
        x1 = random.randint(0, len(g1)-1)
        x2 = random.randint(0, len(g2)-1)
        g3 = np.concatenate((g1[x1:], g2[x2:]))
        if len(g3) > len(g1):
            g3 = g3[0:len(g1)]
        return g3

    @staticmethod
    def point_mutate(genome, rate, amount):
        new_genome = copy.copy(genome)
        for gene in new_genome:
            if random.random() < rate:
                gene[0] = random.randint(0, 1)
                if gene[0] == 0:
                    gene[1] = random.randint(1, 10)
                    gene[2] = random.randint(1, 300)
                elif gene[0] == 4:
                    gene[1] = round(random.uniform(0, 1),1)
                    gene[2] = 0
                else:
                    gene[1] = random.randint(2, 300)
                    gene[2] = 0
        return new_genome

    @staticmethod
    def shrink_mutate(genome, rate):
        if len(genome) == 1:
            return copy.copy(genome)
        if random.random() < rate:
            ind = random.randint(0, len(genome)-1)
            new_genome = np.delete(genome, ind, 0)
            return new_genome
        else:
            return copy.copy(genome)

    @staticmethod
    def grow_mutate(genome, rate):
        if random.random() < rate:
            gene = GA_ML.get_random_layer()
            print(gene)
            new_genome = copy.copy(genome)
            print(new_genome)
            new_genome.append(gene)
            print(new_genome)
            return new_genome
        else:
            return genome


    @staticmethod
    def getMinValue(fits):
        new_fits = copy.copy(fits)
        length = len(new_fits)
        new_fits.sort()
        minValue = new_fits[0]
        secondMinValue = new_fits[1]
        return minValue, secondMinValue

    @staticmethod
    def selectParent(fits):
        minValue, secondMinValue = GA_ML.getMinValue(fits)
        weight = []
        for f in fits:
            if f == minValue:
                weight.append(60)
            elif f == secondMinValue:
                weight.append(20)
            else:
                weight.append(20/len(fits))
        p1, p2 = random.choices(fits, weights=weight, k=2)
        p1_index = fits.index(p1)
        p2_index = fits.index(p2)
        return p1_index, p2_index
