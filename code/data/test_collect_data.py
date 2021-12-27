import unittest
import config
import collect_data
import praw

class CollectDataTest(unittest.TestCase):

    def setUp(self):
        self.reddit =  praw.Reddit(
            user_agent = "comment extraction",
            client_id = config.client_id,
            client_secret = config.client_secret,
            username = config.username,
            password = config.password
        )

    def test_config_clientId(self):
        self.assertIsNotNone(config.client_id)

    def test_config_clientSecret(self):
        self.assertIsNotNone(config.client_secret)

    def test_config_username(self):
        self.assertIsNotNone(config.username)

    def test_config_password(self):
        self.assertIsNotNone(config.password)

    def test_reddit_instance(self):
        self.assertIsNotNone(self.reddit)

    def test_get_comment(self):
        self.assertIsNotNone(collect_data.get_comment('dz1v5cf'))



if __name__ == '__main__':
    unittest.main()
