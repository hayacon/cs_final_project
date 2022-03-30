import unittest
import ga_ml
import ml_simulation
import random
from transformers import AutoTokenizer
import logging



class TestProject(unittest.TestCase):

    def test_simulation(self):
        print('GA begins ......')
        genes = [
            [[1, 30, 0], [1, 15, 0], [2, 73, 0], [3, 77, 0]],
            [[1, 54, 0], [0, 10, 7], [2, 11, 0], [0, 1, 11], [3, 242, 0]],
            [[2, 2, 0], [3, 10, 0]],
            # [[1, 64, 0], [1, 83, 0], [0, 9, 3], [1, 122, 0], [0, 6, 20], [2, 229, 0]],
            # [[1, 64, 0], [1, 83, 0], [0, 9, 3], [1, 122, 0], [0, 6, 20], [2, 229, 0]],
            # [[0, 3, 11], [0, 8, 15], [3, 38, 0], [1, 161, 0]],
            # [[0, 4, 15], [1, 9, 0], [0, 5, 14], [0, 6, 8], [2, 279, 0], [3, 27, 0], [1, 268, 0]],
            # [[0, 9, 11], [1, 271, 0], [0, 9, 16], [3, 50, 0]],
            # [[1, 15, 0], [1, 291, 0], [3, 189, 0]],
            # [[1, 291, 0], [3, 189, 0]]
            ]

        # genes = []
        # for i in range(10):
        #     num_layers = random.randint(5, 8)
        #     genes.append(ga_ml.GA_ML.random_model(num_layers))
        #number of ga iterations
        logger = logging.Logger('catch_all')

        for iteration in range(5):
            print("==================================================")
            if iteration == 0:
                print('1st Iteration')
            elif iteration == 1:
                print('2nd Iteration')
            elif iteration == 2:
                print('3rd Iteration')
            else:
                print(iteration, 'th iteration.....')
            fits = []
            print(genes)
            for gene in genes:
                print("------------------------------")
                try:
                    new_model = ml_simulation.ML_Model(gene)
                    model = new_model.convert_to_model()
                    ml_model = ml_simulation.ML_simulation(tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT"),
                                                            model = model,
                                                            train_df = 'data/training_set.csv',
                                                            test_df = 'data/test_set.csv')

                    result = ml_model.simulation()
                    print('result => ', result)
                    if result < 0.5:
                        print('gene => ', gene)
                    fits.append(result)
                except Exception as e:
                    logger.exception('Failed............ ' + str(e))
                    print('Not a valid model ==>', gene)
                    fits.append(1)
                print("------------------------------")
            print('Fits :=>', fits)
            new_genes = []
            for i in range(15):
                p1, p2 = ga_ml.GA_ML.selectParent(fits)
                new_g = ga_ml.GA_ML.crossover(genes[p1], genes[p2])
                new_dna = ga_ml.GA_ML.point_mutate(new_g, rate=0.1, amount=0.25)
                new_dna = ga_ml.GA_ML.shrink_mutate(new_dna, rate=0.25)
                new_dna = new_dna.tolist()
                new_dna = ga_ml.GA_ML.grow_mutate(new_dna, rate=0.1)
                new_genes.append(new_dna)

            genes = new_genes
            print('new genes =>', genes)
            print("==================================================")


        self.assertNotEqual(fits[0], 0)

unittest.main()
