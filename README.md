# sentiment
messing around with sentiment analysis from the rotten tomatoes dataset

The approach I took here was an <a href=https://en.wikipedia.org/wiki/N-gram>n-gram</a> <a href=https://en.wikipedia.org/wiki/Naive_Bayes_classifier>naive bayesian model</a>.  The code uses the probabilistic bayesian methods to predict the likelihood of sentiments given the n-grams of the incoming document, which are built with the <b>generate_n_grams</b> method. This allows us to overcome SOME of the normally hidden latent contextual information in sentences when just using a tfidf and BOW (bag of words) approach to NLP.
