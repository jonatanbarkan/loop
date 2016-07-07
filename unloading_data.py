import numpy as np


class Unload(object):
    def __init__(self, matrix_size, vector_size=0):
        self.matrix_size = matrix_size
        self.vector_size = vector_size

    def check_matrix(self, sample):
        if sample.shape == self.matrix_size:
            return True
        return False

    def check_vector(self, vector):
        if vector.shape == self.vector_size:
            return True
        return False

    def check_type(self, matrix):
        print(type(matrix))

    def unload(self, sample):
        return np.ravel(sample)
