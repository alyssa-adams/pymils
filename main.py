# Alyssa M Adams, Sept 2020

from pybdm import BDM
import time
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-muted')
from networkx.drawing.nx_agraph import graphviz_layout

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

                    # check if the number of columns or rows (whichever is going to be deleted) is bigger than 4
                    # (pybdm doesn't measure anything smaller than 4x4)
                    num_rows, num_cols = start_image.shape

                    if choice == 0:
                        n_objects = num_rows
                        n_other = num_cols
                    else:
                        n_objects = num_cols
                        n_other = num_rows

                    # if so, then delete an object
                    if n_objects >= 4 and n_other >= 4:

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

                        # check if resulting image is at least 4x4
                        num_rows, num_cols = resulting_image.shape
                        if num_rows >= 4 and num_cols >= 4:

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
                            bdms[str(resulting_image)] = {'bdm': bdm, 'state': str(mapped_resulting_image)}

                        else:  # skip because isn't large enough
                            continue

        return bdms

    def subset_state_space(self, subsets, subset_bdms, image_bdm):

        '''
        Returns a graph with vertices as matrices and edges as single-deletion possible transitions
        :param subsets: dict of lists of lists (index of matrices)
        :return: nx graph
        '''

        g = nx.DiGraph()

        # go through all the levels of subsets and add edges to network
        for step in subsets.keys():
            for start_image in subsets[step].keys():
                for choice in subsets[step][start_image].keys():
                    for resulting_image in subsets[step][start_image][choice]:

                        # check if resulting image is at least 4x4
                        num_rows, num_cols = resulting_image.shape
                        if num_rows >= 4 and num_cols >= 4:

                            # check to see if this is the first one, because need to add initial image edges
                            if step == 0:
                                root_node = str(np.array(image_bdm[0]))
                                root_bdm = image_bdm[1]
                                out_node = subset_bdms[str(resulting_image)]['state']
                                out_bdm = subset_bdms[str(resulting_image)]['bdm']

                                g.add_edge(root_node, out_node)
                                g.add_node(root_node, weight=root_bdm)
                                g.add_node(out_node, weight=out_bdm)

                            else:
                                root_node = subset_bdms[str(start_image)]['state']
                                root_bdm = subset_bdms[str(start_image)]['bdm']
                                out_node = subset_bdms[str(resulting_image)]['state']
                                out_bdm = subset_bdms[str(resulting_image)]['bdm']

                                g.add_edge(root_node, out_node)
                                g.add_node(root_node, weight=root_bdm)
                                g.add_node(out_node, weight=out_bdm)

        return g

    def all_path_bdms(self, statespace):

        """
        Gets all the possible paths from initial image to each leaf and gets the BDM difference at each step
        :param statespace: a nx graph
        :return: a dict of {(rootnode, leafnode): path: bdm_diffs}
        """

        paths = {}

        # get a list of all leaf nodes
        leaf_nodes = [x for x in statespace.nodes() if statespace.out_degree(x) == 0 and statespace.in_degree(x) > 0]
        root_node = [n for n, d in statespace.in_degree() if d==0][0]

        # loop through all the leaf nodes
        for leaf in leaf_nodes:

            paths[(root_node, leaf)] = {}

            all_a_to_b_paths = nx.all_simple_paths(statespace, source=root_node, target=leaf)

            for path in all_a_to_b_paths:
                bdms = [statespace.nodes[node]['weight'] for node in path]
                diffs = np.diff(bdms)

                paths[(root_node, leaf)][str(path)] = diffs

        return paths

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


if __name__ == '__main__':

    # --------------------------------------------------------------
    # Experiment 1: Make the complete network of subsets and time it
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # Experiment 2: What do all the possible paths look like?
    # --------------------------------------------------------------

    # make the images
    size = 7
    number = 5
    PyMILS = PyMILS()

    images = make_images(size, number)

    for image in images:

        # make all possible subsets
        # TODO: Randomly sample branches at each "choice" (run the same image multiple times)
        # TODO: don't go any smaller than 4 rows and edges, the BDM will be 0
        t0 = time.time()
        subsets = PyMILS.image_subsets_complete(image)
        dt = time.time() - t0
        print('Subsets: ' + str(dt))

        # map the data onto this indexed state space and measure bdm
        t0 = time.time()
        subset_bdms = PyMILS.batch_image_bdms(subsets, image)
        dt = time.time() - t0
        print('BDMs: ' + str(dt))

        # get the state space network
        t0 = time.time()
        bdm_tool = PyMILS.init_bdm()
        image_bdm = (image, PyMILS.image_bdm(np.array(image), bdm_tool))
        state_space = PyMILS.subset_state_space(subsets, subset_bdms, image_bdm)
        dt = time.time() - t0
        print('State space network: ' + str(dt))

        # which "kind of paths" through the network preserve the BDM the best?
        # get all paths from start to cutoff size
        t0 = time.time()
        bdm_paths = PyMILS.all_path_bdms(state_space)
        dt = time.time() - t0
        print('Measuring paths: ' + str(dt))

        # plot these differences
        t0 = time.time()

        lines = []

        for ab in bdm_paths.keys():
            for path in bdm_paths[ab]:
                line = bdm_paths[ab][path]
                line = np.cumsum(line)
                lines.append(line)

        [plt.plot(list(range(len(line))), line, linewidth=0.5) for line in lines]
        plt.show()

        dt = time.time() - t0
        print('Plot networks: ' + str(dt))
