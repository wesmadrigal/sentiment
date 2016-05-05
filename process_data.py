#!/usr/bin/env python
import csv

def convert_train(fname):
    """ turns rotten tomatos dataset into a bag of words """
    reader = csv.reader(open(fname, 'r'), delimiter='\t')
    labels = reader.next()
    rows = []
    try:
        while True:
            rows.append(reader.next())
    except StopIteration, e:
        pass
    not_training = rows[0:5000]
    rows = rows[5000:]
    sentiments = [ str(i) for i in range(5) ]
    data_map = {
            'total_words' : 0,
            'bags': {
                k : {
                    'total' : 0,
                    'words' : {}
                    }
                for k in sentiments
                }
            }
    for row in rows:
        sentiment = row[3]
        words = row[2].split(' ')
        for word in words:
            data_map['total_words'] += 1
            if word not in data_map['bags'][sentiment]['words'].keys():
                data_map['bags'][sentiment]['words'][word] = 1
            elif word in data_map['bags'][sentiment]['words'].keys():
                data_map['bags'][sentiment]['words'][word] += 1
            data_map['bags'][sentiment]['total'] += 1
    return data_map, not_training


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


def generate_n_grams(fname, n=3):
    """ 
    generates an N-Gram data structure
    out of the rotten tomatoes dataset
    for better sentiment estimation
    """

    reader = csv.reader(open(fname, 'r'), delimiter='\t')
    labels = reader.next()
    rows = []
    try:
        while True:
            rows.append(reader.next())
    except StopIteration, e:
        pass

    rows = rows[0: (len(rows) - (len(rows)/4))]

    keys = map(lambda x: str(x), range(5))

    sentiment_sentences = {
            k : [
                r[2].lower().split(' ') for r in rows
                if r[3] == k
                ]
            for k in keys
            }
    # structure to keep the N-Grams in

    data = {
            'total_words' : 0
            }
    its = 0
    for sentiment in sentiment_sentences:
        its += 1
        print its
        data[sentiment] = {
                'total' : 0,
                'word_counts' : {}
                }
        for sentence in sentiment_sentences[sentiment]:
            its += 1
            print its
            for idx in range(len(sentence)):
                word = sentence[idx]

                data['total_words'] += 1
                its += 1
                print its
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
                if idx > 3:
                    trigram_back = sentence[idx-3]


                if word not in data[sentiment]['word_counts'].keys():
                    data[sentiment]['word_counts'][word] = {
                            'count' : 1,
                            'n_grams' : {
                                'unigrams' : {},
                                'bigrams' : {},
                                'trigrams' : {}
                            }
                    }
                else:
                    data[sentiment]['word_counts'][word]['count'] += 1
 
                # N-Gram unigram stuff
                if unigram_front:
                    if unigram_front not in data[sentiment]['word_counts'][word]['n_grams']['unigrams'].keys():
                        data[sentiment]['word_counts'][word]['n_grams']['unigrams'][unigram_front] = 1
                    else:
                        data[sentiment]['word_counts'][word]['n_grams']['unigrams'][unigram_front] += 1

                if unigram_back:
                    if unigram_back not in data[sentiment]['word_counts'][word]['n_grams']['unigrams'].keys():
                        data[sentiment]['word_counts'][word]['n_grams']['unigrams'][unigram_back] = 1
                    else:
                        data[sentiment]['word_counts'][word]['n_grams']['unigrams'][unigram_back] += 1


                # N-Gram bigram stuff
                if bigram_front:
                    if bigram_front not in data[sentiment]['word_counts'][word]['n_grams']['bigrams'].keys():
                        data[sentiment]['word_counts'][word]['n_grams']['bigrams'][bigram_front] = 1
                    else:
                        data[sentiment]['word_counts'][word]['n_grams']['bigrams'][bigram_front] += 1

                if bigram_back:
                    if bigram_back not in data[sentiment]['word_counts'][word]['n_grams']['bigrams'].keys():
                        data[sentiment]['word_counts'][word]['n_grams']['bigrams'][bigram_back] = 1
                    else:
                        data[sentiment]['word_counts'][word]['n_grams']['bigrams'][bigram_back] += 1


                # N-Gram trigram
                if trigram_front:
                    if trigram_front not in data[sentiment]['word_counts'][word]['n_grams']['trigrams'].keys():
                        data[sentiment]['word_counts'][word]['n_grams']['trigrams'][trigram_front] = 1
                    else:
                        data[sentiment]['word_counts'][word]['n_grams']['trigrams'][trigram_front] += 1

                if trigram_back:
                    if trigram_back not in data[sentiment]['word_counts'][word]['n_grams']['trigrams'].keys():
                        data[sentiment]['word_counts'][word]['n_grams']['trigrams'][trigram_back] = 1
                    else:
                        data[sentiment]['word_counts'][word]['n_grams']['trigrams'][trigram_back] += 1

    return data       



def weigh_sentiment_ngrams(obj, message):
    """
    takes an object returned from the above
    ngram model and weights the potential
    sentiments that could be possible

    Approach:

    P(sentiment | message) = P(message | sentiment) * P(sentiment) / P(message)
    """

    priors = { }
    posteriors = { }

    # assign the prior counts
    total_words = obj['total_words']
    del obj['total_words']
    for sentiment in obj.keys():
        sentiment_total = 0
        for word in obj[sentiment]['word_counts'].keys():
            sentiment_total += obj[sentiment]['word_counts'][word]['count']
        priors[sentiment] = {
                'count' : sentiment_total,
                'prior' : 0.0
                }
    # assign the prior probabilities
    for sentiment in obj.keys():
        priors[sentiment]['prior'] = float(priors[sentiment]['count']) / float(total_words)


    # message total probability
    # Summation [ P(word1|sentiment) * P(word2|sentiment) ] * P(sentiment) + [ P(word1|sentiment2) * P(word2|sentiment2) ] * P(sentiment2) ...
    message = message.lower().split(' ')
    p_message = float(sum([
        reduce(lambda x, y: x*y, [ 
            float( float(obj[k]['word_counts'][word]['count']) / float(priors[k]['count']) ) * priors[k]['prior']
            for word in message
            if word in obj[k]['word_counts'].keys()
            and obj[k]['word_counts'][word]['count'] > 0
        ])
        for k in priors.keys()
        ]))


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
