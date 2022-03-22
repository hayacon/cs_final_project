import unittest
import numpy as np
import ga_ml

class TestGaMl(unittest.TestCase):

    def setUp(self):
        pass

    def test_class_exist(self):
        self.assertIsNotNone(ga_ml.GA_ML())

    def test_random_layer_exist(self):
        random_layer = ga_ml.GA_ML.get_random_layer()
        print(random_layer)
        self.assertIsNotNone(random_layer)

    def test_crossover_exist(self):
        g1 = [[1, 0, 0], [2, 20, 0], [0, 20, 2]]
        g2 = [[1, 0, 0], [2, 20, 0], [0, 20, 2]]
        self.assertIsNotNone(ga_ml.GA_ML.crossover(g1, g2))

    def test_crossover(self):
        g1 = [[1, 0, 0], [2, 20, 0], [0, 20, 2]]
        g2 = [[1, 0, 0], [2, 20, 0], [0, 20, 2]]
        for i in range(10):
            g3 = ga_ml.GA_ML.crossover(g1, g2)
            self.assertGreater(len(g3), 0)

    def test_point_mutate(self):
        g1 = np.array([[1, 0, 0], [2, 20, 0], [0, 20, 2]])
        self.assertIsNotNone(ga_ml.GA_ML.point_mutate(g1, rate = 1, amount = 0.25))

    def test_shrink_mutate(self):
        g1 = np.array([[1, 0, 0], [2, 20, 0], [0, 20, 2]])
        self.assertIsNotNone(ga_ml.GA_ML.shrink_mutate(g1, 1.0))

    def test_grow_mutate(self):
        g1 = [[1, 30, 0], [2, 20, 0], [0, 20, 2]]
        self.assertIsNotNone(ga_ml.GA_ML.grow_mutate(g1, 0.5))

    def test_getMinValue_exist(self):
        fit = [0.3, 0.21, 0.04, 0.08, 0.1]
        self.assertIsNotNone(ga_ml.GA_ML.getMinValue(fit))

    def test_getMinValue(self):
        fit = [0.3, 0.21, 0.04, 0.08, 0.1]
        minValue, secondMinValue = ga_ml.GA_ML.getMinValue(fit)
        self.assertEqual(minValue, 0.04)
        self.assertEqual(secondMinValue, 0.08)

    def test_selectParent_exist(self):
        fit = [0.3, 0.21, 0.04, 0.08, 0.1]
        self.assertIsNotNone(ga_ml.GA_ML.selectParent(fit))

    def test_selectParent(self):
        fit = [0.3, 0.21, 0.04, 0.08, 0.1]
        p1_index, p2_index = ga_ml.GA_ML.selectParent(fit)
        self.assertTrue(type(p1_index) is int)

    def test_selectParent2(self):
        fit = [0.3, 0.21, 0.04, 0.08, 0.1]
        p1_index, p2_index = ga_ml.GA_ML.selectParent(fit)
        self.assertEqual(p1_index, 2)

unittest.main()
