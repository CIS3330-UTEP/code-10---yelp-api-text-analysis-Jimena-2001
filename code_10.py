from yelpapi import YelpAPI
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords



api_key ="s-KiqfkRtpeXJXXgVmEMKRof7o55zbOdfSzMt35Rm7Sr1Aa8k8UdW9I8_KU_zQ_qcl4MhBI1srTLx-2YSQ1OLqquCnR1kCRnJ6NkVS7ijagiZapLrmwhsN2t8kk2ZHYx"

yelp_api = YelpAPI(api_key)
search_term = "paella"
location_term = "San Antonio, Tx"
search_results = yelp_api.search_query(term=search_term, location=location_term,sort_by='rating', limit=20, offset=20)
print(search_results)
#^^ prints the list of 20 businesses that sells paella in san antonio

#list of Id and price
results_df = pd.DataFrame.from_dict(search_results['businesses'])
print(results_df)
results_df.to_csv("yelpapi_businesses_results.csv")
#request apporoach#

##business search- yelpapi reviews##
id_for_reviews = "toro-kitchen-bar-san-antonio-2"
review_response = yelp_api.reviews_query(id=(id_for_reviews))
#reviews list
print(review_response)
for review in review_response['reviews']:
    print(review['text'])

#yelpapi approach#
id_for_reviews = "toro-kitchen-bar-san-antonio-2"
review_response = yelp_api.reviews_query(id=(id_for_reviews))
print(review_response)
for review in review_response['reviews']:
    print(review['text'])
results_df = pd.DataFrame.from_dict(review_response['reviews'])
print(results_df)
results_df.to_csv(f"{id_for_reviews}_request_reviews_results.csv")

for text in results_df['text']:
    token = nltk.word_tokenize(text)
    tags = nltk.pos_tag(token)
    print(tags)
    for tag in tags:
        if tag[1]=='JJ' or tag[1] =='JJS' or tag[1] =='NN':
            print(tag[0])
print('\n')

analyser= SentimentIntensityAnalyzer()
for review in results_df['text']:
    sentiment_score = analyser.polarity_scores(review)
    print(review)
    print('\n')
    print(sentiment_score)


