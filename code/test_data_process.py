import unittest
import data_process
import pandas as pd
import data_analysis_functions as da

class TestDataProcess(unittest.TestCase):

    def setUp(self):
        self.file = 'data/reddit_dataset.csv'

    # def test_data_vectorization(self):
    #     self.assertIsNotNone(data_process.data_vectorization(self.file))
    #
    # def test_data_vectorization_return_type(self):
    #     expect = pd.core.frame.DataFrame
    #     result = type(data_process.data_vectorization(self.file))
    #     self.assertEqual(result, expect)

    def test_test_train_dataset(self):
        df = da.importData(self.file)
        clean_df = da.cleanDf(df)
        reddit_df = data_process.textConvert(clean_df, 'comment', 'cleaned_comment')
        self.assertIsNotNone(data_process.test_train_dataset(reddit_df))

unittest.main()
