"""
Collect data from Reddit using Reddit API. All comment IDs are provided from original dataset.
"""
# package to use Reddit api
import praw
#import configuration information
import config
import pandas as pd #🐼

#initialize reddit instance with necessary information to use Reddit API
reddit =  praw.Reddit(
    user_agent = "comment extraction",
    client_id = config.client_id,
    client_secret = config.client_secret,
    username = config.username,
    password = config.password
)

# comment = reddit.comment(id="e7v029x")
# print(type(comment.body))
# print(comment.body)

#function to get cooment from Reddit with comment id
def get_comment(id):
    try:
        comment = reddit.comment(id=id)
        return comment.body
    except:
        return "this comment is no longer availble"


#create a new empty pandas dataframe
colunm_name = ["id", "comment", "score"]
final_df = pd.DataFrame(columns = colunm_name)

#read csv file and iterate through each rows
df = pd.read_csv('sample_input_data.csv')
for index, row in df.iterrows():
    comment = get_comment(row['comment'])
    final_df = final_df.append({"id": row["k_id"], "comment":comment, "score":row["Score"]},
                    verify_integrity = True,
                    ignore_index =True,
                    )
    print(str(index) + comment)

print(final_df.head())

final_df.to_csv("reddit_dataset.csv")
