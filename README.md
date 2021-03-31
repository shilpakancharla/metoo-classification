# Tracking Changes in #MeToo Movement with Deep Learning

## Data Description

Our raw data consists of tweets with the hashtag #MeToo from the years of 2017, 2018, and 2019 (links below). We take these datasets and modify them such that we keep only the year the tweet was created in, and the modified text of the tweet (removing hashtags, mentions, retweets, stopwords, emojis, hyperlinks, etc.). The preprocessing can be found in `Preprocessing.ipynb`, and additional helper functions can be found in `custom_regex.py`, `modify_df.py`, and `sentence_processing.py`. The final, cleaned dataset is saved in `processed_data`.

### Links to datasets
* **350K #MeToo Tweets** (October 2017): https://data.world/rdeeds/350k-metoo-tweets
* **390K #MeToo Tweets** (November 2017 - December 2017): https://data.world/balexturner/390-000-metoo-tweets
* **695K Hatred on Twitter During #MeToo Movement** (September 2018 - February 2019): https://www.kaggle.com/rahulgoel1106/hatred-on-twitter-during-metoo-movement
* **15K #MeToo Tweets** (October 2019): https://www.kaggle.com/mohamadalhasan/metoo-tweets-dataset
