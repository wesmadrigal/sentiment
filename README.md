# sentiment
messing around with sentiment analysis from the rotten tomatoes dataset

The approach I took here was an n-gram bayesian model.  The code uses naive bayesian methods to predict the likelihood of sentiments given the n-grams of the incoming document. This allows us to overcome SOME of the normally hidden latent contextual information in sentences when just using a tfidf and BOW (bag of words) approach to NLP.
