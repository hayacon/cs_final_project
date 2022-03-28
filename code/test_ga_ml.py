import unittest
import numpy as np
import ga_ml

class TestGaMl(unittest.TestCase):

    def setUp(self):
        pass

    def test_class_exist(self):
        self.assertIsNotNone(ga_ml.GA_ML())

    def test_random_layer(self):
        random_layer = ga_ml.GA_ML.get_random_layer()
        print(random_layer)
        self.assertIsNotNone(random_layer)

    def test_reandom_model(self):
        model = ga_ml.GA_ML.random_model(4)
        print(model)
        self.assertIsNotNone(model)

    def test_crossover_exist(self):
        g1 = [[1, 0, 0], [2, 20, 0], [0, 20, 2]]
        g2 = [[1, 0, 0], [2, 20, 0], [0, 20, 2]]
        self.assertIsNotNone(ga_ml.GA_ML.crossover(g1, g2))

    def test_crossover(self):
        g1 = [[1, 0, 0], [2, 20, 0], [0, 20, 2]]
        g2 = [[1, 0, 0], [2, 20, 0], [0, 20, 2]]
        # for i in range(10):
        g3 = ga_ml.GA_ML.crossover(g1, g2)
        # print('g1 : ', g1)
        # print('g2 : ', g2)
        # print('g3 : ', g3)
        self.assertGreater(len(g3), 0)

    def test_crossover1(self):
        g1 = [[0, 9, 11], [1, 271, 0], [4, 0.3, 0], [0, 9, 16], [3, 50, 0], [4, 0.7, 0]]
        g2 = [[1, 15, 0], [1, 291, 0], [3, 189, 0]]
        # for i in range(10):
        g3 = ga_ml.GA_ML.crossover(g1, g2)
        # print('g1 : ', g1, len(g1))
        # print('g2 : ', g2, len(g2))
        # print('g3 : ', g3, len(g3))
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

    # def test_selectParent2(self):
    #     fit = [0.3, 0.21, 0.04, 0.08, 0.1]
    #     p1_index, p2_index = ga_ml.GA_ML.selectParent(fit)
    #     self.assertEqual(p1_index, 2)

    # def test_convert_array_to_list(self):
    #     a = [array([[1., 54., 0.],[0., 10.,7.],[1., 131.,0.],[2., 11.,0.],[0., 1.,11.],[3., 242., 0.]]), array([[1.,54. , 0.],[1. ,177. ,0. ],[4. , 0.4, 0.],[2. , 11. , 0.],[0. , 1. , 11.],[3. , 242. ,  0.]])]
    #     self.assertIsNotNone(ga_ml.GA_ML.convert_tolist(a))

    def test_convert_to_int_exist(self):
        a = [[[4.0, 0.2, 0.0], [4.0, 0.3, 0.0], [2.0, 280.0, 0.0], [0.0, 6.0, 161.0], [4.0, 0.1, 0.0]], [[1.0, 54.0, 0.0], [0.0, 10.0, 7.0], [4.0, 0.4, 0.0], [2.0, 11.0, 0.0], [0.0, 1.0, 11.0], [3.0, 242.0, 0.0]], [[1.0, 291.0, 0.0], [3.0, 189.0, 0.0], [1.0, 151.0, 0.0], [4.0, 0.4, 0.0], [2.0, 11.0, 0.0], [0.0, 1.0, 11.0]], [[2, 11, 0], [0, 1, 11], [3, 242, 0], [2, 11, 0], [0, 1, 11], [3, 242, 0]], [[4.0, 0.4, 0.0], [2.0, 11.0, 0.0], [0.0, 1.0, 11.0], [3.0, 242.0, 0.0], [3.0, 242.0, 0.0]]]

        self.assertIsNotNone(ga_ml.GA_ML.convert_toint(a))

    # def test_convert_to_int_exist(self):
    #     a = [[[2.0, 280.0, 0.0], [0.0, 6.0, 161.0]], [[1.0, 54.0, 0.0], [0.0, 10.0, 7.0], [2.0, 11.0, 0.0], [0.0, 1.0, 11.0], [3.0, 242.0, 0.0]], [[1.0, 291.0, 0.0], [3.0, 189.0, 0.0], [1.0, 151.0, 0.0], [2.0, 11.0, 0.0], [0.0, 1.0, 11.0]], [[2, 11, 0], [0, 1, 11], [3, 242, 0], [2, 11, 0], [0, 1, 11], [3, 242, 0]], [ [2.0, 11.0, 0.0], [0.0, 1.0, 11.0], [3.0, 242.0, 0.0], [3.0, 242.0, 0.0]]]
    #
    #     want = [[[2, 280, 0], [0, 6, 161]], [[1, 54, 0], [0, 10, 7], [2, 11, 0], [0, 1, 11], [3, 242, 0]], [[1, 291, 0], [3, 189, 0], [1, 151, 0], [2, 11, 0], [0, 1, 11]], [[2, 11, 0], [0, 1, 11], [3, 242, 0], [2, 11, 0], [0, 1, 11], [3, 242, 0]], [ [2, 11, 0], [0, 1, 11], [3, 242, 0], [3, 242, 0]]]
    #
    #     self.assertIsNotNone(ga_ml.GA_ML.convert_toint(a))



    def test_generating_process(self):
        fits = [0.1, 0.07, 0.01, 0.03, 0.02]
        genes = []
        genes.append(ga_ml.GA_ML.random_model(2))
        genes.append(ga_ml.GA_ML.random_model(3))
        genes.append(ga_ml.GA_ML.random_model(5))
        genes.append(ga_ml.GA_ML.random_model(4))
        genes.append(ga_ml.GA_ML.random_model(3))

        new_genes = []
        for i in range(len(genes)):
            p1, p2 = ga_ml.GA_ML.selectParent(fits)
            new_g = ga_ml.GA_ML.crossover(genes[p1], genes[p2])
            new_dna = ga_ml.GA_ML.point_mutate(new_g, rate=0.1, amount=0.25)
            new_dna = ga_ml.GA_ML.shrink_mutate(new_dna, rate=0.25)
            # new_dna = ga_ml.GA_ML.grow_mutate(new_dna, rate=0.1)
            new_dna = new_dna.tolist()
            new_genes.append(new_dna)
        print('new genes', new_genes)
        print(type(new_genes))
        print(type(new_genes[0]))

unittest.main()
