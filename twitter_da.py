import tweepy
import os
import pandas as pd
from textblob import TextBlob
from collections import Counter
from pprint import pprint
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime

date = datetime.today().strftime('%Y%m%d')
month = datetime.today().strftime('%Y%m')
time = datetime.now().strftime('%H%M')

load_dotenv()
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("BEARER_TOKEN")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

query = os.getenv("QUERY")
tweets_cnt = int(os.getenv("TWEETS_CNT"))

client = tweepy.Client(bearer_token=bearer_token,
                    consumer_key=consumer_key,
                    consumer_secret=consumer_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret)


#####################################################################################################################
### main function
def twitter_da(event, context):

    ## ACTION: Data processing
    #  collect tweets
    tweets_info = searchTweets(client, query, tweets_cnt)

    # data cleaning
    tweets_text_split, tweets_hashtags, tweets_at = Tweets_Clean_Data(tweets_info)


    ## ACTION: Top 10 popular words
    top_words = DA_popular_words(tweets_text_split)
    text_top_words = 'Top 10 popular words of #' + query.split()[0] + ' on Twitter today (' + date + '):\n' + top_words
    client.create_tweet(text = text_top_words)


    ## ACTION: Top 10 popular hashtags
    top_hashtags = DA_popular_hashtags(tweets_hashtags)
    text_top_hashtags =  'Top 10 popular hashtags of #' + query.split()[0] + ' on Twitter today (' + date + '):\n' + top_hashtags
    client.create_tweet(text = text_top_hashtags)


    ## ACTION: Top 10 popular users
    top_users = DA_popular_users(tweets_at)
    text_top_users =  'Top 10 mentioned users of #' + query.split()[0] + ' on Twitter today (' + date + '):\n' + top_users
    client.create_tweet(text = text_top_users)
    

    ## ACTION: Top 10 popular tweets
    # data transform to DataFrame
    df_tweets_info = Tweets_Core_Data(tweets_info)

    top_tweets_id = DA_popular_tweets(df_tweets_info)
    for i in range(len(top_tweets_id)):
        text =  'Top ' + str(i+1) + ' popular tweet of #' + query.split()[0] + ' today (' + date + '):'
        client.create_tweet(text=text, quote_tweet_id=top_tweets_id[i])       
        

    ## ACTION: tweets sentimate
    # add sentiment analysis column
    df_tweets_info_tb = DA_Tweets_Sentiment(df_tweets_info)

    tweets_senti_des = DA_Sentiment_Describe(df_tweets_info_tb)
    
    pol_mean = str(round(tweets_senti_des['pol_mean'],2))
    text_pol_mean = 'The public sentiment of #' + query.split()[0] + ' on Twitter today (' + date + ') is ' + pol_mean
    client.create_tweet(text = text_pol_mean)


    ## ACTION: make a word cloud
    Word_Cloud(tweets_text_split)


###
#####################################################################################################################





###
#####################################################################################################################
### function: collect tweets
def searchTweets(client, query, tweets_cnt):

    tweet_fields = ['author_id', 'created_at', 'lang', 'possibly_sensitive', 'source', 'geo', 'entities', 'public_metrics', 'conversation_id', 'referenced_tweets']
    user_fields = ['id', 'name', 'username', 'created_at','profile_image_url','public_metrics']
    expansions = ['author_id', 'referenced_tweets.id', 'geo.place_id', 'attachments.media_keys', 'in_reply_to_user_id']
    start_time = None
    end_time = None

    tweets = tweepy.Paginator(client.search_recent_tweets, 
                    query=query, 
                    tweet_fields=tweet_fields, 
                    user_fields=user_fields,
                    expansions=expansions,
                    start_time=start_time,
                    end_time=end_time,
                    max_results=100).flatten(limit=tweets_cnt)
   
    tweets_info = []

    for tweet in tweets: 
        tweets_info.append(tweet.data)

    ### the following piece of code is cited from TwitterCollector by Gene Moo Lee, Jaecheol Park and Xiaoke Zhang    
    result = {}
    result['collection_type'] = 'recent post'
    result['query'] = query
    result['tweet_cnt'] = len(tweets_info)
    result['tweets'] = tweets_info
    ###

    return result



### function: clean tweets data (core data)
def Tweets_Core_Data(tweets_info):

    tweets_core = []

    for i in range(0, len(tweets_info['tweets'])):
        tweet_time = tweets_info['tweets'][i]['created_at']
        tweet_text = tweets_info['tweets'][i]['text']
        tweet_id = tweets_info['tweets'][i]['id']
        author_id = tweets_info['tweets'][i]['author_id']

        # public_metrics
        pm_impression_count = tweets_info['tweets'][i]['public_metrics']['impression_count']
        pm_like_count = tweets_info['tweets'][i]['public_metrics']['like_count']
        pm_quote_count = tweets_info['tweets'][i]['public_metrics']['quote_count']
        pm_reply_count = tweets_info['tweets'][i]['public_metrics']['reply_count']
        pm_retweet_count = tweets_info['tweets'][i]['public_metrics']['retweet_count']

        record = [tweet_time, tweet_text, tweet_id, author_id, pm_impression_count, pm_like_count, pm_quote_count, pm_reply_count, pm_retweet_count]
        tweets_core.append(record)
    
    # transform to dataframe
    df_tweets_core = pd.DataFrame(tweets_core, columns =['Time', 'Text', 'Tweet_id', 'Author_id','impression_count', 'like_count', 'quote_count', 'reply_count', 'retweet_count'])

    return df_tweets_core



### function: clean tweets data (abandon stopwords)
def Tweets_Clean_Data(tweets_info):

    stop_words_orig = ['our', 'those', 'my', 'them', 'which', "should've", 'y', 'yours', 'each', 'mustn', 'own', 're', 'more', 'only', 'but', 'before', 'himself', "doesn't", 'at', 's', 'll', 'didn', 'while', 'd', 'whom', 'against', 'm', 'by', 'to', 'some', "she's", 'about', 'am', 'until', 'were', 'for', 'into', "didn't", 'ma', 'having', 'off', "shouldn't", 'this', 'who', "shan't", 'his', 'been', "hasn't", "hadn't", 'did', "needn't", 'these', 'won', "wasn't", 'now', 'wasn', 'myself', 'same', 'further', 'both', 'out', 't', "mustn't", 'from', 'over', 'few', 'aren', "won't", 'again', "that'll", 'ain', 'if', 'can', 'no', 'as', 'had', 'a', 'the', 'above', 'other', 'she', 'all', 'should', 'because', "it's", "couldn't", 'very', 'and', 'how', 'their', 'where', 'most', 'doing', 'was', 'being', 'him', 'through', 'hadn', 'up', 'there', 'when', 'themselves', 'once', 'itself', 'why', 'wouldn', 'your', 'is', 'weren', 'are', 'down', 'then', 'such', 'herself', 'too', 'will', "aren't", "haven't", "weren't", 'we', 'you', 'nor', 'has', 'in', 'with', "don't", 'between', 'he', 'below', 'under', "isn't", 'shouldn', 'o', "wouldn't", 'have', 'they', 'on', 'do', 'yourself', 'yourselves', 'its', 'here', "you'll", "you've", "you're", 've', 'couldn', "you'd", 'does', 'haven', 'i', 'needn', 'or', 'hers', 'hasn', 'any', 'doesn', 'than', 'me', 'theirs', 'after', 'ourselves', 'isn', 'don', 'that', 'just', 'ours', 'during', 'of', 'mightn', 'be', 'shan', 'it', 'her', 'an', 'what', 'not', "mightn't", 'so']
    stop_words_add = ['https', 'amp','amp;','`','#','i’ve','she’s','would','ha']
    stop_words_symbol = ["’","/",".","#",'``']
    stop_words = stop_words_orig + stop_words_add + stop_words_symbol

    tweets_text_orig = []
    tweets_text_s_1 = []
    tweets_text_s_2 = []
    tweets_text_split = []
    tweets_hashtags = []
    tweets_at = []

    for i in range(0, len(tweets_info['tweets'])):
        tweets_text_orig.append(tweets_info['tweets'][i]['text'])

    for i in range(len(tweets_text_orig)):
        tweets_text_s_1.append(tweets_text_orig[i].split())

    for i in tweets_text_s_1:
        tweets_text_s_2.extend(i)

    for w in tweets_text_s_2:
        if w.lower() not in stop_words and len(w)>1:
            if w.isalnum() ==1:
                tweets_text_split.append(w.lower())
            elif '@' in w:
                tweets_at.append(w)
            elif "#" in w:
                tweets_hashtags.append(w.lower())          

    return tweets_text_split, tweets_hashtags, tweets_at


### function: find the most popular words
def DA_popular_words(words):

    words_count = Counter(words)
    words_count_top10 = words_count.most_common(10)
    output =  To_Post_Format(words_count_top10)

    return output



## function: find most popular hashtags
def DA_popular_hashtags(hashtags):

    hashtags_count = Counter(hashtags)
    hashtags_count_top10 = hashtags_count.most_common(10)
    output = To_Post_Format(hashtags_count_top10)

    return output



### function: find most mentioned users
def DA_popular_users(users):

    users_count = Counter(users)
    users_count_top10 = users_count.most_common(10)
    output = To_Post_Format(users_count_top10)

    return output



### function: find most popular tweets
def DA_popular_tweets(df_tweets_info):

    df_top_popular_tweets = df_tweets_info.sort_values(by=['impression_count'], ascending=False)
    output = df_top_popular_tweets[:3]['Tweet_id'].tolist()

    return output



### function: calculate the sentimate
def DA_Tweets_Sentiment(df_tweets_info):

    tweets_pol = []
    tweets_sub = []

    for i in range(len(df_tweets_info)):
        tweets_pol.append(TextBlob(df_tweets_info.iloc[i]['Text']).sentiment.polarity)
        tweets_sub.append(TextBlob(df_tweets_info.iloc[i]['Text']).sentiment.subjectivity)

    df_tweets_info['Text_pol'] = tweets_pol
    df_tweets_info['Text_sub'] = tweets_sub

    return df_tweets_info


### function: the description of sentimate
def DA_Sentiment_Describe(df_tweets_info):
    
    pol_mean = df_tweets_info['Text_pol'].mean()    
    sub_mean = df_tweets_info['Text_sub'].mean()

    pol_top_id = df_tweets_info.loc[df_tweets_info.Text_pol == df_tweets_info['Text_pol'].max()].Tweet_id
    sub_top_id = df_tweets_info.loc[df_tweets_info.Text_sub == df_tweets_info['Text_sub'].max()].Tweet_id

    pol_buttom_id = df_tweets_info.loc[df_tweets_info.Text_pol == df_tweets_info['Text_pol'].min()].Tweet_id
    sub_buttom_id = df_tweets_info.loc[df_tweets_info.Text_sub == df_tweets_info['Text_sub'].min()].Tweet_id

    output = {}
    output['pol_mean'] = pol_mean
    output['sub_mean'] = sub_mean
    output['pol_top_id'] = pol_top_id
    output['sub_top_id'] = sub_top_id
    output['pol_buttom_id'] = pol_buttom_id
    #output['sub_buttom_id'] = sub_buttom_id

    return output



### function: make a word cloud
def Word_Cloud(words):

    words_input = ' '.join(words)
    wordcloud = WordCloud(collocations=False, width=800, height=400).generate(words_input)

    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()



### function: transform the format can be posted
def To_Post_Format(list):
    post_format_1 = []
    for i in range(0, len(list)):
        post_format_1.append([i+1, ', ', list[i][0], '\n'])
    
    post_format_1 = sum(post_format_1, [])

    post_format = ''
    post_format = ''.join([str(elem) for elem in post_format_1])

    return post_format



def check(input):
    a = type(input)
    b = len(input)
    print('type: ' + str(a))
    print('len: ' + str(b))