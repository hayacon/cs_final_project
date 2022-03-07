import unittest
import data_analysis_functions as da
import pandas as pd #🐼
import os

class TestDataAnalysis(unittest.TestCase):

    def setUp(self):
        self.csv_file = "./data/reddit_dataset.csv"
        self.df = None

    def test_importData(self):
        name, extention = os.path.splitext(self.csv_file)
        self.assertEqual(extention, '.csv')
        self.assertIsNotNone(da.importData(self.csv_file))

    def test_importData_returnType(self):
        expect = pd.core.frame.DataFrame
        result = type(da.importData(self.csv_file))
        self.assertEqual(result, expect)

    def test_cleanDf(self):
        df = da.importData(self.csv_file)
        self.assertIsNotNone(da.cleanDf(df))

    def test_cleandDf_returnType(self):
        df = da.importData(self.csv_file)
        expect = pd.core.frame.DataFrame
        result = type(da.cleanDf(df))
        self.assertEqual(result, expect)

    def test_cleanDf_returnData(self):
        df = da.importData(self.csv_file)
        result_df = da.cleanDf(df)
        result = True
        for index, row in df.iterrows():
            if row['comment'] == '[removed]' or row['comment'] == '[deleted]' or row['comment'] == 'this comment is no longer availble':
                result = False
                break
        self.assertTrue(result)

    def test_dataDistribution(self):
        df = da.importData(self.csv_file)
        self.assertIsNotNone(da.dataDistribution(df))

    def test_dataDistribution_returnType(self):
        df = da.importData(self.csv_file)
        expect = pd.core.frame.DataFrame
        result = type(da.dataDistribution(df))
        self.assertEqual(result, expect)

    def test_dataDistribution_returnDf(self):
        df = da.importData(self.csv_file)
        dist_df = da.dataDistribution(df)
        col_name = dist_df.columns
        self.assertEqual(list(col_name), ['-1 ~ -0.75', '-0.75 ~ -0.5', '-0.5 ~ -0.25', '-0.25 ~ 0', '0 ~ 0.25','0.25 ~ 0.5', '0.5 ~ 0.75', '0.75 ~ 1'])

    def test_cleanText(self):
        self.assertIsNotNone(da.cleanText('hi'))

    def test_cleanText_returnType(self):
        text = "I'm currently studying my BSc degree in https://www.coursera.org/ provided by @UoL"
        clean_text = da.cleanText(text)
        self.assertTrue(isinstance(clean_text, list))

    def test_cleanText_returnString(self):
        text = "I'm currently studying my BSc degree in https://www.coursera.org/ provided by @UoL"
        expect = ['currently', 'studying', 'bsc', 'degree', 'provided', 'uol']
        result = da.cleanText(text)
        self.assertEqual(result, expect)


if __name__ == '__main__':
    unittest.main()
