# Alyssa M Adams, Sept 2020

from pybdm import BDM
import random
import networkx as nx
import numpy as np
from operator import itemgetter
from itertools import groupby

import pybdm
pybdm.options.set(raise_if_zero=False)


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
        #max_size = 10
        #n_choices = dict(zip(range(2, max_size), range(1, max_size*2, 2)))
        #n_choices = n_choices[size]

        start_images = [image]
        choice_number = 0

        while len(start_images) > 0:

            subsets[choice_number] = {}

            # for each image in the start nodes at choice number t
            for start_image in start_images:

                subsets[choice_number][str(start_image.tolist())] = {}

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

                        subsets[choice_number][str(start_image.tolist())][choice] = final_images

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
                            bdms[str(resulting_image.tolist())] = {'bdm': bdm, 'state': str(mapped_resulting_image.tolist())}

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
                                root_node = str(image_bdm[0])
                                root_bdm = image_bdm[1]
                                out_node = subset_bdms[str(resulting_image.tolist())]['state']
                                out_bdm = subset_bdms[str(resulting_image.tolist())]['bdm']

                            else:
                                root_node = subset_bdms[start_image]['state']
                                root_bdm = subset_bdms[start_image]['bdm']
                                out_node = subset_bdms[str(resulting_image.tolist())]['state']
                                out_bdm = subset_bdms[str(resulting_image.tolist())]['bdm']

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

                # change path to list of lists, not nd arrays (for loading back in from string)
                paths[(root_node, leaf)][str(path)] = diffs

        return paths

    def mils(self, img, min_size, bdm_tool, sampling, chunk_size):  # TODO: Add min complexity cutoff?

        '''
        Takes an image, randomly picks if it will remove a row or column, then removes a random one of that choice
        :param image: list of lists: matrix of binary values that is an image
        :param min_size: 0-1 Minimum final image size (cannot be smaller than 4 though)
        :param sampling: float 0-1, percent of rows/columns to sample possible next steps
        :return: final image
        '''

        # part 1 is very fast for small images but way too slow for larger ones
        # but parts 2 3 tell us that we can just randomly pick paths so we don't need to calculate all possible paths

        # 0 = row, 1 = column

        # get size information of original image
        size = np.array(img).shape
        min_pixels = int(min_size*min(size))
        rounded_aspect_ratio = int(max(size)/min(size))

        # get coords of bottom right pixel of upper left quadrant (for experimental verification)
        mid_coordinates = (round(size[0]/2), round(size[1]/2))

        # pre-compute the row and column deletion order to preserve the aspect ratio
        # row = 0, column = 1
        min_rows = int(min_size*size[0])
        n_row_deletions = int((size[0]-min_rows)/4)
        min_columns = int(min_size*size[1])
        n_column_deletions = int((size[1]-min_columns)/4)
        total_deletions = n_row_deletions + n_column_deletions

        # change symbol every rounded_aspect_ratio th element
        # determine which element to start with
        if rounded_aspect_ratio != 1 and n_row_deletions > n_column_deletions:  # if there are more rows to delete
            start_choice = 0
        elif rounded_aspect_ratio == 1:  # 1:1 aspect ratio can start with rows by default
            start_choice = 0
        else:  # not 1:1 and more column deletions, start with columns
            start_choice = 1

        # this is the list the algorithm will follow to make deletions and preserve the aspect ratio
        deletions = []
        for i in range(total_deletions):
            if i % (rounded_aspect_ratio+1) == 0:  # append, then flip which one we add
                if start_choice == 1:
                    choice = 0
                else:
                    choice = 1
                deletions.append(choice)
            else:
                deletions.append(start_choice)

        # starting image complexity
        bdm_start = self.image_bdm(np.array(img), bdm_tool)

        # make sure the image is large enough
        if min_pixels < 4:
            print("min_size gives less than 4 pixels! Pick a larger size. (BDM doesn't measure anything smaller than a 4x4 matrix)")
            quit()


        # make sure the image is still big enough
        # while it is, do this loop and take out rows and columns
        #while size[0] > min_pixels and size[1] > min_pixels:

        for choice in deletions:

            # randomly pick sampling proportion of indexes from choice
            n_indices = size[choice]
            sampling_indices = random.sample(range(n_indices), int(sampling*n_indices))

            # loop through each potential index to remove and pick the one with the closest bdm value
            bdm_sample_results = []

            for i in sampling_indices:

                # remove the index
                if choice == 0:  # remove rows in chunks of 4
                    if i > len(img)-chunk_size:  # check for boundary wrap-around
                        img_sample = img[i % chunk_size:i]
                    else:
                        img_sample = img[:i] + img[i+chunk_size:]
                else:  # removes columns
                    if i > len(img[0]) - chunk_size:  # check for boundary wrap-around
                        img_sample = list(map(lambda x: x[i % chunk_size:i], img))
                    else:
                        img_sample = list(map(lambda x: x[:i] + x[i + chunk_size:], img))

                # get the resulting bdm
                bdm_sample = self.image_bdm(np.array(img_sample), bdm_tool)
                bdm_sample_results.append((i, bdm_sample))

            # pick the index with the closest bdm value (if multiple, just pick the first one)
            bdm_sample_results = list(map(lambda x: (x[0], abs(x[1]-bdm_start)), bdm_sample_results))
            bdm_sample_results = sorted(bdm_sample_results, key=lambda x: x[1])

            # if there are more than one indices that result in a change of 0, just pick a random
            grouped_indices = [[i for i, j in temp] for key, temp in groupby(bdm_sample_results, key=itemgetter(1))]
            index_to_remove = random.choice(grouped_indices[0])

            # remove the index
            if choice == 0:
                if index_to_remove > len(img) - chunk_size:  # check for boundary wrap-around
                    img = img[index_to_remove % chunk_size:index_to_remove]
                else:
                    img = img[:index_to_remove] + img[index_to_remove + chunk_size:]
            else:
                if index_to_remove > len(img[0]) - chunk_size:  # check for boundary wrap-around
                    img = list(map(lambda x: x[index_to_remove % chunk_size:index_to_remove], img))
                else:
                    img = list(map(lambda x: x[:index_to_remove] + x[index_to_remove + chunk_size:], img))

            # reset the size to check if size is big enough
            size = np.array(img).shape

            # see if the "center" moved. don't forget about the chunk size!
            # 0 = row, 1 = column
            if choice == 0:
                if index_to_remove < mid_coordinates[0]:
                    # check to see if the chunk overlaps with the middle
                    overlap = mid_coordinates[0] - index_to_remove
                    if overlap < chunk_size:
                        overlap = overlap
                    else:
                        overlap = chunk_size
                    mid_coordinates = (mid_coordinates[0]-overlap, mid_coordinates[1])
            else:
                if index_to_remove < mid_coordinates[1]:
                    # check to see if the chunk overlaps with the middle
                    overlap = mid_coordinates[1] - index_to_remove
                    if overlap < chunk_size:
                        overlap = overlap
                    else:
                        overlap = chunk_size
                    mid_coordinates = (mid_coordinates[0], mid_coordinates[1]-overlap)

        return img, mid_coordinates
