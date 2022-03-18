import unittest
import numpy as np
import ga_ml

class TestGaMl(unittest.TestCase):

    def setUp(self):
        pass

    def test_class_exist(self):
        self.assertIsNotNone(ga_ml.GA_ML())

    def test_crossover(self):
        g1 = [[1], [2], [3]]
        g2 = [[4], [5], [6]]
        self.assertIsNotNone(ga_ml.GA_ML.crossover(g1, g2))

    def test_point_mutate(self):
        g1 = np.array[[1.0], [2.0]]
        self.assertIsNotNone(ga_ml.GA_ML.point_mutate(genome, rate, amount))

    def test_shrink_mutate(self):
        g1 = np.array[[1.0], [2.0]]
        self.assertIsNotNone(ga_ml.GA_ML.shrink_mutate(genome, rate))

    def test_grow_mutate(self):
        g1 = np.array[[1.0], [2.0]]
        self.assertIsNotNone(ga_ml.GA_ML.grow_mutate(genome, rate))

unittest.main()
