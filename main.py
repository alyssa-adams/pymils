# Alyssa M Adams, Sept 2020

from pybdm import BDM
import time
import random
import networkx as nx
import numpy as np

import pybdm
pybdm.options.set(raise_if_zero=False)


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

    def init_bdm(self):

        '''
        Initialize the bdm class for 2 dimensions and 2 symbols
        :return: bdm class
        '''

        bdm = BDM(ndim=2)

        return bdm

    def image_bdm(self, image, bdm_tool):

        '''
        Returns the BDM of a matrix
        :param image: List of lists
        :param bdm_tool: List of lists
        :return: float
        '''

        bdm = bdm_tool.bdm(image)

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

        # some quick math of how many choices are made for a matrix of this size
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
                        n_other = num_cols
                    else:
                        n_objects = num_cols
                        n_other = num_rows

                    # if so, then delete an object
                    if n_objects > 1 and n_other > 0:

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

    def batch_image_bdms(self, subsets, image):

        '''
        Measures the BDMs of all possible subsets
        :param subsets: dict of lists of lists (index of matrices)
        :param image: Matrix, this image gets to be mapped onto the indexed subsets
        :return: dict for the index of bdms
        '''

        # make the dict of matrix coords to map back onto
        image_size = len(image)
        cells = image_size**2
        coords = []
        for i in range(image_size):
            for j in range(image_size):
                coords.append((i, j))
        coords = dict(zip(range(cells), coords))

        # load in the BDM object
        bdm_tool = self.init_bdm()
        bdms = {}

        # measure all the BDMs
        for step in subsets.keys():
            for start_image in subsets[step].keys():
                for choice in subsets[step][start_image].keys():
                    for resulting_image in subsets[step][start_image][choice]:

                        # check if resulting image is at least one cell
                        num_rows, num_cols = resulting_image.shape
                        if num_rows + num_cols > 0:

                            # map image onto index
                            mapped_resulting_image = []

                            for row in resulting_image:
                                mapped_row = []
                                for cell in row:
                                    data_coords = coords[cell]
                                    mapped_cell = image[data_coords[0]][data_coords[1]]
                                    mapped_row.append(mapped_cell)
                                mapped_resulting_image.append(mapped_row)

                            mapped_resulting_image = np.array(mapped_resulting_image)
                            bdm = self.image_bdm(mapped_resulting_image, bdm_tool)
                            bdms[str(mapped_resulting_image)] = bdm

                        else:  # skip because isn't large enough
                            continue

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

    # --------------------------------------------------------------
    # Experiment 1: Make the complete network of subsets and time it
    # --------------------------------------------------------------

    # make the images
    size = 6
    number = 5
    PyMILS = PyMILS()

    images = make_images(size, number)

    for image in images:

        # make all possible subsets
        # TODO: Randomly sample branches at each "choice" (run the same image multiple times)
        t0 = time.time()
        subsets = PyMILS.image_subsets_complete(image)
        dt = time.time() - t0
        print('Subsets: ' + str(dt))

        # map the data onto this indexed state space
        t0 = time.time()
        subset_bdms = PyMILS.batch_image_bdms(subsets, image)
        dt = time.time() - t0
        print('BDMs: ' + str(dt))

        # turn this into a network
        t0 = time.time()
        #state_space = PyMILS.subset_state_space(subsets)
        #dt = time.time() - t0
        #print(dt)
