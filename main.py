# Alyssa M Adams, Sept 2020

import pybdm
import time
import random
import networkx as nx
import numpy as np


# make some sample images: random, blank, patterned, cellular automata, photos, and half/half mixes

def make_images(size, number):

    '''
    Make a bunch of images of a particular size x size
    :param size: int
    :return: list of matrices
    '''

    images = [[[random.choice([0, 1]) for _ in range(size)] for __ in range(size)] for ___ in range(number)]

    return images


# the MILS algorithm:
# 1. measure BDM of matrix
# 2. Remove a row or column that keeps the BDM to the closest value
# 3. Repeat

# Try with random deletion
# Then try SVM and random trees

class PyMILS:

    # image := matrix of binary values, a list of lists

    def image_bdm(self, image):

        '''
        Returns the BDM of a matrix
        :param image: List of lists
        :return: float
        '''

        bdm = image

        return bdm

    def image_subsets_complete(self, image):

        '''
        Makes all possible subsets of a particular graph
        :param image: list of lists
        :return: dict of lists of lists (index of matrices)
        '''

        subsets = []

        # index the cells for dev purposes
        size = len(image)
        image = list(zip(*(iter(range(size**2)),) * size))
        image = np.array(image)

        # at each step we need to take a row or a cell while there are 2 or more of each option
        # if there's only one row or cell, can't take, so choice is to take the one that is bigger than 1
        # Make this choice 1 3 5 7 9 ... times for sizes 2 3 4 5 6 ... respectively (I counted!).

        n_choices = dict(zip(range(2, 10), range(1, 20, 2)))
        n_choices = n_choices[size]

        # for each choice that is made
        for choice_n in range(n_choices):

            # These are the two possible choices
            choices = [0, 1]  # row and column, respectively

            # for each choice
            for choice in choices:

                # check if the number of columns or rows (whichever is going to be deleted) is bigger than 1
                num_rows, num_cols = image.shape

                if choice == 0:
                    n_objects = num_rows
                else:
                    n_objects = num_cols

                # if so, then delete an edge
                if n_objects > 1:

                    for index_to_remove in range(n_objects):

                        image = np.delete(image, index_to_remove, axis=choice)
                        subsets.append(image)

                else:
                    continue


        subsets = []
        subsets = {}  # index to link bdm information

        return subsets

    def batch_image_bdms(self, subsets):

        '''
        Measures the BDMs of all possible subsets
        :param subsets: dict of lists of lists (index of matrices)
        :return: dict for the index of bdms
        '''

        bdms = {}

        for i in subsets.keys():
            bdms[i] = self.image_bdm(subsets[i])

        return bdms

    def subset_state_space(self, subsets):

        '''
        Returns a graph with vertices as matrices and edges as single-deletion possible transitions
        :param subsets: dict of lists of lists (index of matrices)
        :return: nx graph
        '''

        g = nx.Graph()

        return g

    """
    def remove_object(self, method, object, image):

        '''
        Returns the image with the object removed
        :param method: string: random selection, svm, random tree, etc.
        :param object: string: row or column
        :param image: list of lists: matrix of binary values
        :return: list of lists: matrix with that object removed
        '''

        new_image = image

        return image

    def mils(self, method, image):

        '''
        Applies a method of removing rows and columns from a matrix to get a minimal matrix with maximally
        preserved information
        :param method: string: random selection, svm, random tree, etc.
        :param image: list of lists: matrix of binary values that is an image
        :return: list of lists: the reduced matrix
        '''

        while value < threshold:

            bdm = self.image_bdm(image)
            object = random.choice(['row', 'column'])
            new_image = self.remove_object(method, object, image)
            new_bdm =
    """

# Test to see how long these functions will take to run!

if __name__ == '__main__':

    # make the images
    size = 4
    number = 10
    images = make_images(size, number)

    PyMILS = PyMILS()

    for image in images:
        t0 = time.time()
        subsets = PyMILS.image_subsets_complete(image)
        dt = time.time() - t0
        print(dt)
        t0 = time.time()
        subset_bdms = PyMILS.batch_image_bdms(subsets)
        dt = time.time() - t0
        print(dt)
        t0 = time.time()
        state_space = PyMILS.subset_state_space(subsets)
        dt = time.time() - t0
        print(dt)
