import unittest
import ml_simulation
from transformers import AutoTokenizer
import data_analysis_functions as da

class TestMlSimulation(unittest.TestCase):

    def setUp(self):
        self.gene = [[0, 3, 20], [2, 2, 0], [3, 10, 0], [4, 0.5, 0]]
        self.tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
        self.trainDf = 'data/training_set.csv'
        self.testDf = 'data/test_set.csv'

    def test_ML_Model(self):
        self.assertIsNotNone(ml_simulation.ML_Model)

    def test_convwert_to_model(self):
        model = ml_simulation.ML_Model(self.gene)
        self.assertIsNotNone(model.convert_to_model())

    def test_convert_to_model_return_type(self):
        model = ml_simulation.ML_Model(self.gene)
        result = model.convert_to_model()
        self.assertEqual(str(type(result)), "<class 'keras.engine.sequential.Sequential'>")

    def test_ML_simulation(self):
        self.assertIsNotNone(ml_simulation.ML_simulation)

    # def test_filter_long_description(self):
        # train_df = da.importData(self.trainDf)
        # self.assertIsNotNone(ml_simulation.ML_simulation.filter_long_descriptions(self.tokenizer, train_df.encoded_comment.tolist(), 300))

    # def test_short_description(self):
    #     m = ml_simulation.ML_Model(self.gene)
    #     model = m.convert_to_model()
    #     train_df = da.importData(self.trainDf)
    #     test_df = da.importData(self.testDf)
    #     simulation = ml_simulation.ML_simulation(self.tokenizer, model, train_df, test_df)
    #     self.assertIsNotNone(simulation.short_description(train_df))

    def test_simulation(self):
        m = ml_simulation.ML_Model(self.gene)
        model = m.convert_to_model()
        train_df = da.importData(self.trainDf)
        test_df = da.importData(self.testDf)
        # print(train_df)
        simulation = ml_simulation.ML_simulation(self.tokenizer, model, train_df, test_df)
        result = simulation.simulation()
        # print(result)
        self.assertIsNotNone(result)


unittest.main()
