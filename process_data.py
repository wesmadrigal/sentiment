#!/usr/bin/env python
import re
import csv
import time
from stopwords import stopwords

def parse_train_data(fname, split=0.9):
    """
    Parses training data from rotten tomatoes dataset
    train.tsv file, removes stopwords, and splits based
    on the parameterized split float

    Parameters
    ----------
    fname : str of filename containig the training data
    split : float of percentage of our data to train vs. keep as test data

    Returns
    ---------
    data : tuple of (train, test) data    
    """
    reader = csv.reader(open(fname, 'r'), delimiter='\t')
    rows = []
    label = reader.next()
    try:
        while True:
            rows.append(reader.next())
    except StopIteration:
        pass
    newrows = []
    for row in rows:
        msg = row[2]
        parsed_msg = filter(lambda x: x not in stopwords and len(re.findall(r'[a-z]+', x)) > 0, msg.lower().split())
        if parsed_msg != []:
            row[2] = ' '.join(parsed_msg)
            newrows.append( row )

    print "{0} % of original rows left after removing stopwords".format(float(len(newrows))/float(len(rows))*100)
    return (newrows[0:int(len(newrows)*split)], newrows[int(len(newrows)*split):])

def k_folds(fname, k=10):
    """
    Parameters
    ----------
    fname : str of training data filename
    k : int of number of folds to run

    Returns
    ---------
    results : list of tuples with (accuracy, time)
    """
    results = []
    test_step = 1.0 / float(k)    
    train, test = parse_train_data(fname, split=1.0 / float(k))
    alldata = train + test
    for z in range(k):
        if z != 0:
            idx_start = int((z * test_step) * len(alldata))
            idx_stop = int(((z+1) * test_step) * len(alldata))
            test = alldata[idx_start:idx_stop]
            train = alldata[0:idx_start] + alldata[idx_stop:]
        start = time.time()
        print "Running k-folds iteration %d" % z
        this_obj, this_total = generate_n_grams(train)
        acc = measure_accuracy_order(this_obj, this_total, test)
        stop = time.time()
        acc_perc = float(sum(acc)) / float(len(acc))
        runtime = stop - start
        results.append( (acc_perc, runtime) )        
    return results


def weigh_sentiment(message, obj):
    """ takes a review and an object returned from the above as a parameter
    we'll be using a Naive Bayesian Classifier to weight the methods
    Example:
        P(sentiment | message) = P(message | sentiment ) * P(sentiment) / P(message)

    """
    words = message.split(' ')
    sentiments = ['0', '1', '2', '3', '4']
    p_message_given_sentiments = { }
    # for each sentiment
    # weigh P(message | sentiment)
    for k in sentiments:
        vals = []
        for w in words:
            try:
                val = float(obj['bags'][k]['words'][w]) / float(obj['bags'][k]['total'])
            except:
                val = 1.0 / float(obj['bags'][k]['total'])
            vals.append( val )
        p_message_given_sentiments[k] = float(sum(vals))

    p_sentiments = {
           k : float(obj['bags'][k]['total']) / float(obj['total_words'])
           for k in sentiments 
           }
    
    p_message = []
    # altered 11/19/2014
    # had total probability wrongly
    # calculating before
    for k in sentiments:
        this_k = []
        for w in words:
            try:
                p_w_given_k = float(obj['bags'][k]['words'][w]) / float(obj['bags'][k]['total'])
            except Exception, e:
                print e.message
                p_w_given_k = 1.0 / float(obj['bags'][k]['total'])
            #val = p_w_given_k * p_sentiments[k]
            val = p_w_given_k
            this_k.append( val )
        p_message_given_k = reduce(map(lambda x, y: x*y, this_k)) * p_sentiments[k]
        #p_message.append(sum(this_k))
        p_message.append(p_message_given_k)
    p_message = sum(p_message)

    probabilities = {
           k : ( p_message_given_sentiments[k] * p_sentiments[k] ) / p_message
           for k in sentiments 
           }
    return probabilities



def weigh_sentiment_markovian(message, obj):
    """
    using a mixture of Markovian and Bayesian methods
    to weight our message's sentiment
    """
    words = message.split(' ')

    words_and_sentiment = {}


def generate_n_grams(train, n=3):
    """ 
    generates an N-Gram data structure
    out of the rotten tomatoes dataset
    for better sentiment estimation

    Parameters
    ----------
    train : list of training rows from the dataset stopwords should already be removed
    
    n : int for number of n-grams to use, defaults to 3

    Returns
    -------
    ds : dict data structure
    """
    keys = map(lambda x: str(x), range(5))

    sentiment_sentences = {
            k : [ r[2].lower().split() for r in train if r[3] == k ]
            for k in keys
            }
    # structure to keep the N-Grams in
    data = {
            'total_words' : 0            
            }
    total_words = 0
    data = {
            str(k) : {
                'total' : 0,
                'word_counts' : {}
                }
            for k in range(5)
            }
    its = 0
    for row in train:
        sentiment = row[3]
        sentence = row[2].lower().split()
        for idx in range(len(sentence)):
            word = sentence[idx]
            total_words += 1
            data[sentiment]['total'] += 1
            unigram_front = None
            unigram_back = None
            bigram_front = None
            bigram_back = None
            trigram_front = None
            trigram_back = None
            if idx < len(sentence)-1:
                unigram_front = sentence[idx+1]
            if idx > 0:
                unigram_back = sentence[idx-1]

            if idx < len(sentence)-2:
                bigram_front = sentence[idx+2]
            if idx > 1:
                bigram_back = sentence[idx-2]

            if idx < len(sentence)-3:
                trigram_front = sentence[idx+3]
            if idx > 2:
                trigram_back = sentence[idx-3]
            if not data[sentiment]['word_counts'].get(word):
                data[sentiment]['word_counts'][word] = {
                        'count' : 1,
                        'n_grams' : {
                            'unigrams' : {
                                'front' : {},
                                'back' : {}
                                },
                            'bigrams' : {
                                'front' : {},
                                'back' : {}
                                },
                            'trigrams' : {
                                'front' : {},
                                'back' : {}
                                }
                        }
                }
            else:
                data[sentiment]['word_counts'][word]['count'] += 1

            # N-Gram unigram stuff
            if unigram_front:
                if not data[sentiment]['word_counts'][word]['n_grams']['unigrams']['front'].get(unigram_front):
                    data[sentiment]['word_counts'][word]['n_grams']['unigrams']['front'][unigram_front] = 1
                else:
                    data[sentiment]['word_counts'][word]['n_grams']['unigrams']['front'][unigram_front] += 1

            if unigram_back:
                if not data[sentiment]['word_counts'][word]['n_grams']['unigrams']['back'].get(unigram_back):
                    data[sentiment]['word_counts'][word]['n_grams']['unigrams']['back'][unigram_back] = 1
                else:
                    data[sentiment]['word_counts'][word]['n_grams']['unigrams']['back'][unigram_back] += 1
            # N-Gram bigram stuff
            if bigram_front:
                if not data[sentiment]['word_counts'][word]['n_grams']['bigrams']['front'].get(bigram_front, None):
                    data[sentiment]['word_counts'][word]['n_grams']['bigrams']['front'][bigram_front] = 1
                else:
                    data[sentiment]['word_counts'][word]['n_grams']['bigrams']['front'][bigram_front] += 1

            if bigram_back:
                if not data[sentiment]['word_counts'][word]['n_grams']['bigrams']['back'].get(bigram_back, None):
                    data[sentiment]['word_counts'][word]['n_grams']['bigrams']['back'][bigram_back] = 1
                else:
                    data[sentiment]['word_counts'][word]['n_grams']['bigrams']['back'][bigram_back] += 1
            # N-Gram trigram stuff
            if trigram_front:
                if not data[sentiment]['word_counts'][word]['n_grams']['trigrams']['front'].get(trigram_front, None):
                    data[sentiment]['word_counts'][word]['n_grams']['trigrams']['front'][trigram_front] = 1
                else:
                    data[sentiment]['word_counts'][word]['n_grams']['trigrams']['front'][trigram_front] += 1

            if trigram_back:
                if not data[sentiment]['word_counts'][word]['n_grams']['trigrams']['back'].get(trigram_back, None):
                    data[sentiment]['word_counts'][word]['n_grams']['trigrams']['back'][trigram_back] = 1
                else:
                    data[sentiment]['word_counts'][word]['n_grams']['trigrams']['back'][trigram_back] += 1

    return data, total_words      



def weigh_sentiment_ngrams(obj, message, total_words):
    """
    takes an object returned from the above
    ngram model and weights the potential
    sentiments that could be possible

    Approach:

    P(sentiment | message) = P(message | sentiment) * P(sentiment) / P(message)
    
    Parameters
    ----------
    obj : dict returned from generate_n_grams method
    message : str of message
    total_words : int of total words in bag of words
    """

    priors = { }
    posteriors = { }

    # assign the prior counts
    #total_words = obj['total_words']
    #del obj['total_words']
    for sentiment in obj.keys():
        sentiment_total = obj[sentiment]['total']
        priors[sentiment] = {
                'count' : sentiment_total,
                'prior' : float(sentiment_total) / float(total_words)
                }

    # message total probability
    # Summation [ P(word1|sentiment) * P(word2|sentiment) ] * P(sentiment) + [ P(word1|sentiment2) * P(word2|sentiment2) ] * P(sentiment2) ...
    message = message.lower().split(' ')
    p_message = 0.0
    for k in priors.keys():
        p_m_given_k = 1.0
        for word in message:
            try:
                p_m_given_k *= ( float(obj[k]['word_counts'][word]['count']) / float(priors[k]['count']) ) * priors[k]['prior']
            except Exception, e:
                p_m_given_k *= ( 1.0 / float(total_words) ) * priors[k]['prior']
        p_message += p_m_given_k

    # weigh the message
    for sentiment in obj.keys():
        p_msg_given_sentiment = 0
        p_sentiment = priors[sentiment]['prior']
        for idx in range(len(message)):
            word = message[idx]
            try:
                p_word_given_sentiment = float(obj[sentiment]['word_counts'][word]['count']) / float(priors[sentiment]['count'])
            except (KeyError, ZeroDivisionError), e:
                # a potentially effective, different approach
                p_word_given_sentiment = 1.0 / float(total_words)
                

            p_msg_given_sentiment += p_word_given_sentiment

            # attempt to add on the n-gram weight
            # if we have a word for the n-gram
            """
            The idea for weighting our n-gram words is as follows:
            if we have an n-gram word W:
                P(ngram-W | sentiment) = C(W|s) / sum( C(W|s1), C(W|s2), C(W|s3), ..., C(W|sn) )

                where C(W) represents the the count of word W as an n-gram
                of the current word in the conditioned sentiment class
            """

            ###############
            # UNIGRAM
            ###############
            # unigram front
            if idx < len(message) - 1:
                unigram_word = message[idx+1]
                try:

                    unigram_counts_s = obj[sentiment]['word_counts'][word]['n_grams']['unigrams'][unigram_word]
                    unigram_counts_all_s = float(sum([
                        obj[k]['word_counts'][word]['n_grams']['unigrams'][unigram_word]
                        for k in obj.keys()
                        if word in obj[k]['word_counts'].keys()
                        and unigram_word in obj[k]['word_counts'][word]['n_grams']['unigrams'].keys()
                        ]))
                    p_unigram_given_sentiment = float(unigram_counts_s) / unigram_counts_all_s

                except Exception, e:
                    print "Didnt have unigram for {0} in sentiment {1}".format(unigram_word, sentiment)
                    p_unigram_given_sentiment = 1.0 / float(total_words)

                p_msg_given_sentiment += p_unigram_given_sentiment

            # unigram back
            if idx > 0:
                unigram_word = message[idx-1]
                try:

                    unigram_counts_s = obj[sentiment]['word_counts'][word]['n_grams']['unigrams'][unigram_word]
                    unigram_counts_all_s = float(sum([
                        obj[k]['word_counts'][word]['n_grams']['unigrams'][unigram_word]
                        for k in obj.keys()
                        if word in obj[k]['word_counts'].keys()
                        and unigram_word in obj[k]['word_counts'][word]['n_grams']['unigrams'].keys()
                        ]))
                    p_unigram_given_sentiment = float(unigram_counts_s) / unigram_counts_all_s

                except Exception, e:
                    print "Didnt have unigram for {0} in sentiment {1}".format(unigram_word, sentiment)
                    p_unigram_given_sentiment = 1.0 / float(total_words)

                p_msg_given_sentiment += p_unigram_given_sentiment


            ################
            # BIGRAM
            ################

            # attempt to add on the n-gram weight
            # if we have a word for the n-gram
            # bigram front
            if idx < len(message) - 2:
                bigram_word = message[idx+1]
                try:
                    bigram_counts_s = obj[sentiment]['word_counts'][word]['n_grams']['bigrams'][bigram_word]
                    bigram_counts_all_s = float(sum([
                        obj[k]['word_counts'][word]['n_grams']['bigrams'][bigram_word]
                        for k in obj.keys()
                        if word in obj[k]['word_counts'].keys()
                        and bigram_word in obj[k]['word_counts'][word]['n_grams']['bigrams'].keys()
                        ]))

                    p_bigram_given_sentiment = float(bigram_counts_s) / bigram_counts_all_s
                except Exception, e:
                    print "Didn't have bigram for {0} in sentiment {1}".format(bigram_word, sentiment)
                    p_bigram_given_sentiment = 1.0 / float(total_words)

                p_msg_given_sentiment += p_bigram_given_sentiment

            # bigram back
            if idx > 1:
                bigram_word = message[idx-2]
                try:
                    bigram_counts_s = obj[sentiment]['word_counts'][word]['n_grams']['bigrams'][bigram_word]
                    bigram_counts_all_s = float(sum([
                        obj[k]['word_counts'][word]['n_grams']['bigrams'][bigram_word]
                        for k in obj.keys()
                        if word in obj[k]['word_counts'].keys()
                        and bigram_word in obj[k]['word_counts'][word]['n_grams']['bigrams'].keys()
                        ]))

                    p_bigram_given_sentiment = float(bigram_counts_s) / bigram_counts_all_s
                except Exception, e:
                    print "Didn't have bigram for {0} in sentiment {1}".format(bigram_word, sentiment)
                    p_bigram_given_sentiment = 1.0 / float(total_words)

                p_msg_given_sentiment += p_bigram_given_sentiment


            #################
            # TRIGRAM
            #################
            if idx < len(message) - 3:
                trigram_word = message[idx+3]
                try:
                    trigram_count_s = obj[sentiment]['word_counts'][word]['n_grams']['trigrams'][trigram_word]
                    trigram_counts_all_s = float(sum([
                        obj[k]['word_counts'][word]['n_grams']['trigrams'][trigram_word]
                        for k in obj.keys()
                        if word in obj[k]['word_counts'].keys()
                        and trigram_word in obj[k]['word_counts'][word]['n_grams']['trigrams'].keys()
                        ]))

                    p_trigram_given_sentiment = float(trigram_count_s) / trigram_counts_all_s

                except Exception, e:
                    print "Didnt have trigram for {0} in sentiment {1}".format(trigram_word, sentiment)
                    p_trigram_given_sentiment = 1.0 / float(total_words)

                p_msg_given_sentiment += p_trigram_given_sentiment


            # trigram back
            if idx > 2:
                trigram_word = message[idx-3]
                try:
                    trigram_count_s = obj[sentiment]['word_counts'][word]['n_grams']['trigrams'][trigram_word]
                    trigram_counts_all_s = float(sum([
                        obj[k]['word_counts'][word]['n_grams']['trigrams'][trigram_word]
                        for k in obj.keys()
                        if word in obj[k]['word_counts'].keys()
                        and trigram_word in obj[k]['word_counts'][word]['n_grams']['trigrams'].keys()
                        ]))

                    p_trigram_given_sentiment = float(trigram_count_s) / trigram_counts_all_s

                except Exception, e:
                    print "Didnt have trigram for {0} in sentiment {1}".format(trigram_word, sentiment)
                    p_trigram_given_sentiment = 1.0 / float(total_words)

                p_msg_given_sentiment += p_trigram_given_sentiment

        # apply Bayes rule
        posterior = ( p_msg_given_sentiment * p_sentiment ) / p_message
        posteriors[sentiment] = posterior
    return posteriors


def find_max(probs):
    """
    @param: probs
    @type: L{dict}

    Takes a probability distribution object
    returned from weigh_sentiment_ngrams
    and returns the maximum likelihood sentiment
    """
    _max = ''
    for k in probs.keys():
        if _max == '':
            _max = k
        elif probs[_max] < probs[k]:
            _max = k
    return _max


def measure_accuracy(ngram_obj, total, test_rows):
    """
    @param: ngram_obj
    @type: L{dict} returned from generate_ngrams
    @param: total
    @type: L{int}
    @param: test_rows
    @type: L{list} of rows of instances
    return L{list} of booleans of correct and incorrect guesses
    """

    measurements = []
    for row in test_rows:
        # make sure we've got the total in the dict
        ngram_obj['total_words'] = total
        message = row[2]
        sentiment = row[3]
        try:
            probs = weigh_sentiment_ngrams(ngram_obj, message)
            max_sentiment = find_max(probs)
            val = 1 if sentiment == max_sentiment else 0
            measurements.append( val )
            print probs
        except Exception, e:
            print e.message
    return measurements

def weigh_sentiment_ngrams_order(obj, message, total_words):
    """
    ORDER MATTERS 
    takes an object returned from the above
    ngram model and weights the potential
    sentiments that could be possible

    Approach:

    P(sentiment | message) = P(message | sentiment) * P(sentiment) / P(message)
    
    Parameters
    ----------
    obj : dict returned from generate_n_grams method
    message : str of message
    total_words : int of total words in bag of words

    Returns
    ---------
    posteriors : dict of posterior probabilities associated with each sentiment class label
    """
    priors = { }
    posteriors = { }

    # assign the prior counts
    for sentiment in obj.keys():
        sentiment_total = obj[sentiment]['total']
        priors[sentiment] = {
                'count' : sentiment_total,
                'prior' : float(sentiment_total) / float(total_words)
                }

    # message total probability
    # Summation [ P(word1|sentiment) * P(word2|sentiment) ] * P(sentiment) + [ P(word1|sentiment2) * P(word2|sentiment2) ] * P(sentiment2) ...
    message = message.lower().split(' ')
    p_message = 0.0
    for k in priors.keys():
        p_m_given_k = 1.0
        for word in message:
            try:
                p_m_given_k *= ( float(obj[k]['word_counts'][word]['count']) / float(priors[k]['count']) ) * priors[k]['prior']
            except Exception, e:
                p_m_given_k *= ( 1.0 / float(total_words) ) * priors[k]['prior']
        p_message += p_m_given_k

    # weigh the message
    for sentiment in obj.keys():
        p_msg_given_sentiment = 1.0
        p_sentiment = priors[sentiment]['prior']
        for idx in range(len(message)):
            word = message[idx]
            try:
                p_word_given_sentiment = float(obj[sentiment]['word_counts'][word]['count']) / float(priors[sentiment]['count'])
            except (KeyError, ZeroDivisionError), e:
                # a potentially effective, different approach
                p_word_given_sentiment = 1.0 / float(total_words)                
            p_msg_given_sentiment *= p_word_given_sentiment

            # attempt to add on the n-gram weight
            # if we have a word for the n-gram
            """
            The idea for weighting our n-gram words is as follows:
            if we have an n-gram word W:
                P(ngram-W | sentiment) = C(W|s) / sum( C(W|s1), C(W|s2), C(W|s3), ..., C(W|sn) )

                where C(W) represents the the count of word W as an n-gram
                of the current word in the conditioned sentiment class
            """
            ###############
            # UNIGRAM
            ###############
            # unigram front
            if idx < len(message) - 1:
                unigram_word = message[idx+1]
                # p(unigram_word | word)
                try:
                    p_unigram_front_given_word = float(obj[sentiment]['word_counts'][word]['n_grams']['unigrams']['front'][unigram_word]) / float(obj[sentiment]['word_counts'][word]['n_grams']['unigrams']['front'].keys().__len__())
                except Exception, e:
                    p_unigram_front_given_word = 1.0 / float(total_words)

                p_msg_given_sentiment *= p_unigram_front_given_word

            # unigram back
            if idx > 0:
                unigram_word = message[idx-1]
                try:
                    p_unigram_back_given_word = float(obj[sentiment]['word_counts'][word]['n_grams']['unigrams']['back'][unigram_word]) / float(obj[sentiment]['word_counts'][word]['n_grams']['unigrams']['back'].keys().__len__())
                except Exception, e:
                    p_unigram_back_given_word = 1.0 / float(total_words)

                p_msg_given_sentiment *= p_unigram_back_given_word

            ################
            # BIGRAM
            ################
            # attempt to add on the n-gram weight
            # if we have a word for the n-gram
            # bigram front
            if idx < len(message) - 2:
                bigram_word = message[idx+2]
                try:
                    p_bigram_front_given_word = float(obj[sentiment]['word_counts'][word]['n_grams']['bigrams']['front'][bigram_word]) / float(obj[sentiment]['word_counts'][word]['n_grams']['bigrams']['front'].keys().__len__())
                except Exception, e:
                    p_bigram_front_given_word = 1.0 / float(total_words)

                p_msg_given_sentiment *= p_bigram_front_given_word

            # bigram back
            if idx > 1:
                bigram_word = message[idx-2]
                try:
                    p_bigram_back_given_word = float(obj[sentiment]['word_counts'][word]['n_grams']['bigrams']['back'][bigram_word]) / float(obj[sentiment]['word_counts'][word]['n_grams']['bigrams']['back'].keys().__len__())
                except Exception, e:
                    p_bigram_back_given_word = 1.0 / float(total_words)

                p_msg_given_sentiment *= p_bigram_back_given_word

            # TRIGRAM
            if idx < len(message)-3:
                trigram_word = message[idx+3]                
                try:
                    trigram_front_len = len(obj[sentiment]['word_counts'][word]['n_grams']['trigrams']['front'].keys())
                    p_trigram_front_given_word = float(obj[sentiment]['word_counts'][word]['n_grams']['trigrams']['front'][trigram_word]) / float(trigram_front_len)
                except KeyError, e:
                    p_trigram_front_given_word = 1.0 / float(total_words)
                p_msg_given_sentiment *= p_trigram_front_given_word
            if idx > 2:
                trigram_word = message[idx-3]
                try:
                    trigram_back_len = len(obj[sentiment]['word_counts'][word]['n_grams']['trigrams']['back'].keys())
                    p_trigram_back_given_word = float(obj[sentiment]['word_counts'][word]['n_grams']['trigrams']['back'][trigram_word]) / float(trigram_back_len)
                except KeyError, e:
                    p_trigram_back_given_word = 1.0 / float(total_words)
                p_msg_given_sentiment *= p_trigram_back_given_word 

        # apply Bayes rule
        posterior = ( p_msg_given_sentiment * p_sentiment ) / p_message
        posteriors[sentiment] = posterior
    return posteriors, priors


def measure_accuracy_order(ngram_obj, total, test_rows):
    """
    @param: ngram_obj
    @type: L{dict} returned from generate_ngrams
    @param: total
    @type: L{int}
    @param: test_rows
    @type: L{list} of rows of instances
    return L{list} of booleans of correct and incorrect guesses
    """

    measurements = []
    for row in test_rows:
        # make sure we've got the total in the dict
        message = row[2]
        sentiment = row[3]
        try:
            probs, priors = weigh_sentiment_ngrams_order(ngram_obj, message, total)
            max_sentiment = find_max(probs)
            val = 1 if sentiment == max_sentiment else 0
            measurements.append( val )
            print probs
        except Exception, e:
            print e.message
    return measurements
