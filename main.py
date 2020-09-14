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

        subsets = {}

        # index the cells for dev purposes
        size = len(image)
        image = list(zip(*(iter(range(size**2)),) * size))
        image = np.array(image)

        # at each step we need to take a row or a cell while there are 2 or more of each option
        # if there's only one row or cell, can't take, so choice is to take the one that is bigger than 1

        max_size = 10
        n_choices = dict(zip(range(2, max_size), range(1, max_size*2, 2)))
        n_choices = n_choices[size]

        start_images = [image]
        choice_number = 0

        while len(start_images) > 0:

            subsets[choice_number] = {}

            # for each image in the start nodes at choice number t
            for start_image in start_images:

                subsets[choice_number][str(start_image)] = {}

                # These are the two possible choices
                choices = [0, 1]  # row and column, respectively

                # for each choice
                for choice in choices:

                    # check if the number of columns or rows (whichever is going to be deleted) is bigger than 1
                    num_rows, num_cols = start_image.shape

                    if choice == 0:
                        n_objects = num_rows
                    else:
                        n_objects = num_cols

                    # if so, then delete an object
                    if n_objects > 1:

                        final_images = []

                        for index_to_remove in range(n_objects):

                            final_image = np.delete(start_image, index_to_remove, axis=choice)

                            # save these
                            final_images.append(final_image)

                        subsets[choice_number][str(start_image)][choice] = final_images

                    else:
                        continue

            # update the choice number and get the set of new start images
            choice_number += 1
            last_level = subsets[list(subsets.keys())[-1]]

            start_images = []

            for key in last_level.keys():

                # grab and merge from both choices at each step
                if 0 in last_level[key].keys():
                    choice_0_results = last_level[key][0]
                else:
                    choice_0_results = []
                if 1 in last_level[key].keys():
                    choice_1_results = last_level[key][1]
                else:
                    choice_1_results = []

                start_images.append(choice_0_results + choice_1_results)

            start_images = [item for sublist in start_images for item in sublist]

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
    size = 8
    number = 5
    PyMILS = PyMILS()

    for s in range(4, size):

        images = make_images(s, number)

        for image in images:

            t0 = time.time()
            subsets = PyMILS.image_subsets_complete(image)
            dt = time.time() - t0
            print((dt, s))

            """
            t0 = time.time()
            subset_bdms = PyMILS.batch_image_bdms(subsets)
            dt = time.time() - t0
            print(dt)
            t0 = time.time()
            state_space = PyMILS.subset_state_space(subsets)
            dt = time.time() - t0
            print(dt)"""
