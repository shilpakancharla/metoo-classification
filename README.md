# Tracking Changes in #MeToo Movement with Deep Learning

## Motivation

Social media provides a unique opportunity to shed light on phenomena that have previously been ignored or swept under the rug. The #MeToo movement began in 2006 when Tarana Burke first coined the phrase, but a viral Twitter post in 2017 served to lift the hashtag into the mainstream accompanied by the public reckoning of several high-profile men. This hashtag allows people to seek solidarity on a much larger scale in calling out the perpetrators of their abuse. We are going to categorize tweets with #MeToo into the years they were posted. In particular, we look at the years 2017, 2018, and 2019.

After compiling a classification model, we plan to use the LIME package to explain the pattern learned by the deep learning model. Such a process allows us to have a direct understanding of the difference the posts of the three years. Specifically, LIME provides us with a linear coefficient to every word that appears in one tweet, and tells us which one of them contributes most in categorizing them to one of three years. In using this approach, we can learn the difference by ourselves and explain them to others in language, which is far more better than a black box machine learning model. We list several observations that have a high possibility in each year, and show why our model categorizes it into such category. Some initial discussions are raised at the end of this report.

## Data Description

Our raw data consists of tweets with the hashtag #MeToo from the years of 2017, 2018, and 2019 (links below). We take these datasets and modify them such that we keep only the year the tweet was created in, and the modified text of the tweet (removing hashtags, mentions, retweets, stopwords, emojis, hyperlinks, etc.). The preprocessing can be found in `Preprocessing.ipynb`, and additional helper functions can be found in `custom_regex.py`, `modify_df.py`, and `sentence_processing.py`.

### Links to datasets
* **350K #MeToo Tweets** (October 2017): https://data.world/rdeeds/350k-metoo-tweets
* **390K #MeToo Tweets** (November 2017 - December 2017): https://data.world/balexturner/390-000-metoo-tweets
* **695K Hatred on Twitter During #MeToo Movement** (September 2018 - February 2019): https://www.kaggle.com/rahulgoel1106/hatred-on-twitter-during-metoo-movement
* **15K #MeToo Tweets** (October 2019): https://www.kaggle.com/mohamadalhasan/metoo-tweets-dataset
