import unittest
import data_analysis_functions as da
import pandas as pd #🐼
import os

class TestDataAnalysis(unittest.TestCase):

    def setUp(self):
        self.csv_file = "./data/reddit_dataset.csv"
        self.df = da.importData(self.csv_file)

    def test_importData(self):
        name, extention = os.path.splitext(self.csv_file)
        self.assertEqual(extention, '.csv')

    def test_importData_returnType(self):
        expect = pd.core.frame.DataFrame
        result = type(da.importData(self.csv_file))
        self.assertEqual(result, expect)

    def test_importData_col(self):
        expect = ['id', 'comment', 'score']
        cols = self.df.columns
        self.assertEqual(list(cols), expect)

    def test_cleanDf(self):
        self.assertIsNotNone(da.cleanDf(self.df))

    def test_cleandDf_returnType(self):
        expect = pd.core.frame.DataFrame
        result = type(da.cleanDf(self.df))
        self.assertEqual(result, expect)

    def test_cleanDf_returnData(self):
        result_df = da.cleanDf(self.df)
        result = True
        for index, row in self.df.iterrows():
            if row['comment'] == '[removed]' or row['comment'] == '[deleted]' or row['comment'] == 'this comment is no longer availble':
                result = False
                break
        self.assertTrue(result)

    def test_dataDistribution(self):
        self.assertIsNotNone(da.dataDistribution(self.df))

    def test_dataDistribution_returnType(self):
        expect = pd.core.frame.DataFrame
        result = type(da.dataDistribution(self.df))
        self.assertEqual(result, expect)

    def test_dataDistribution_returnDf(self):
        dist_df = da.dataDistribution(self.df)
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

    def test_flatten(self):
        self.assertIsNotNone(da.flatten(self.df, 'comment'))

    def test_flatten_returnType(self):
        result = da.flatten(self.df, 'comment')
        self.assertTrue(isinstance(result, list))

    def test_textConvert(self):
        self.assertIsNotNone(da.textConvert(self.df, 'comment', 'cleaned_comment'))

    def test_textConvert_dfCol(self):
        result_df = da.textConvert(self.df, 'comment', 'cleaned_comment')
        columns = result_df.columns
        self.assertEqual(list(columns), ['id', 'comment', 'cleaned_comment', 'score'])

    def test_textConvert_dfColNotDefalut(self):
        result_df = da.textConvert(self.df, 'comment', 'cleaned_comment')
        self.assertNotEqual(result_df['cleaned_comment'][0], 'abc')

    def test_lexical_diversity(self):
        text = ['currently', 'studying', 'bsc', 'degree', 'provided', 'uol']
        self.assertIsNotNone(da.lexical_diversity(text))

    def test_lexical_diversity(self):
        text = ['currently', 'studying', 'bsc', 'degree', 'provided', 'uol']
        vocab, lex = da.lexical_diversity(text)
        self.assertEqual(vocab, 6)
        self.assertEqual(lex, 1)

    def test_plot_lexical_diversity(self):
        self.assertIsNotNone(da.plot_lexical_diversity([0, 1], [0, 1]))

    def test_plot_lexical_diversity_returnTypr(self):
        expect = pd.core.frame.DataFrame
        result = type(da.plot_lexical_diversity([0, 1],[0, 1]))
        self.assertEqual(result, expect)

    def test_frequentDistribution(self):
        self.assertIsNotNone(da.frequentDistribution('hi'))

    def test_frequentDistribution_result(self):
        text = ['Apple', 'Apple', 'MicroSoft', 'Amazon', 'Apple', 'Amazon', 'Google', 'Amazon', 'Apple', 'Google']
        expect = [('Apple', 4), ('Amazon',3), ('Google',2), ('MicroSoft',1)]
        self.assertEqual(expect, da.frequentDistribution(text))

    def test_plot_wordCloud(self):
        self.assertIsNotNone(da.plot_wordCloud('hi'))

if __name__ == '__main__':
    unittest.main()
