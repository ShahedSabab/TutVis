import re
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
import gensim 
import gensim.corpora as corpora
import pyLDAvis.gensim
import numpy
#from gensim.utils import simple_preproess 
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from pprint import pprint
import spacy
import pyLDAvis
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


documents = []
stopWords = []



def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
def remove_stopwords(texts):
    with open(r'''D:\CLoud\Academic\Research\___\Backup\stopwords_en.txt''') as f1:    
        for line in f1:
            stopWords.append(line.strip())
    return [[word for word in simple_preprocess(str(doc)) if word not in stopWords] for doc in texts]

def length_check(texts):
    return [[word for word in simple_preprocess(str(doc)) if len(word)>2] for doc in texts]
    

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_value_mallet(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    mallet_path = 'C:/mallet/bin/mallet'
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def compute_coherence_value_gensim(dictionary, corpus, texts, limit, start, step):
   
    coherence_values = []
    model_list = []
    perplexity_values = []
    for num_topics in range(start, limit, step):        
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
        perplexity = model.log_perplexity(corpus)
        perplexity_values.append(perplexity)
        
    return model_list, coherence_values, perplexity_values

def fileRead(fileNumber): 

    input_file =   r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\4.1 Analysis (using 750 symmetrical data)\1. Feature (Words)\Tf IDF\Input\S('''+repr(fileNumber)+''').txt'''
   

    
    f = open (input_file, 'r',encoding="utf-8",errors='ignore') 
    lines = ''.join(f.readlines())
    lines = lines.lower().strip('\n')
    lines= re.sub('[^A-Za-z\.\,]+', ' ', lines)
    
    return lines

def fileWrite(fileNumber,text,length):
     out_path = r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\4.2 Analyzing with Bigram and trigram (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Output\A'''+repr(fileNumber)+'''_.txt'''
     fileOut =open(out_path,"a",encoding="utf-8")
     fileOut.write("\n"+text)
     fileOut.write("\n"+str(length))
     

def plotCoherence(limit, start, step,coherence_values):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    return x

def plotPerplexity(limit, start, step,perplexity_values):
    y = range(start, limit, step)
    plt.plot(x, perplexity_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity")
    plt.legend(("perplexity_values"), loc='best')
    plt.show()
    return y

def malletmodel2ldamodel(mallet_model, gamma_threshold=0.001, iterations=50):
    """
    Function to convert mallet model to gensim LdaModel. This works by copying the
    training model weights (alpha, beta...) from a trained mallet model into the
    gensim model.

    Args:
    mallet_model : Trained mallet model
    gamma_threshold : To be used for inference in the new LdaModel.
    iterations : number of iterations to be used for inference in the new LdaModel.

    Returns:
    model_gensim : LdaModel instance; copied gensim LdaModel
    """
    model_gensim = gensim.models.ldamodel.LdaModel(
    id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
    alpha=mallet_model.alpha, iterations=iterations,
    eta=mallet_model.word_topics,
    gamma_threshold=gamma_threshold,
    dtype=numpy.float64 # don't loose precision when converting from MALLET
    )
    model_gensim.expElogbeta[:] = mallet_model.wordtopics
    return model_gensim

def format_dominant_topics(ldamodel, corpus, texts):
    global start_doc
    global end_doc
    # Init output
    sent_topics_df = pd.DataFrame()
     
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num)+1, round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    file_name = []
    for x in range(start_doc,end_doc):
        file_name.append(x)
        
    file = pd.Series(file_name)
    sent_topics_df = pd.concat([file, sent_topics_df], axis=1)
    # Add original text to the end of the output
    #contents = pd.Series(texts)
    #sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
   
    return(sent_topics_df)

def format_document_topic_distribution(ldamodel, corpus, num_topic):
    global start_doc
    global end_doc
    # Init output
    topics = []
    doc_topics = []

    for i, row in enumerate(ldamodel[corpus]):
        topics = [prop_topic for j, (topic_num, prop_topic) in enumerate(row)]
        doc_topics.append(topics)
    sent_topics_df = pd.DataFrame.from_records(doc_topics)        
    column_name = []  
    file_name = []
    
    
    for x in range(1,num_topic+1):
        column_name.append("Topic "+str(x))
    for x in range(start_doc,end_doc):
        file_name.append(x)
        
    sent_topics_df.columns = column_name
    file = pd.Series(file_name)
    sent_topics_df = pd.concat([file, sent_topics_df], axis=1)
    return(sent_topics_df)
        # Get the Dominant topic, Perc Contribution and Keywords for each document




start_doc=1 
end_doc=751
 
for x in range(start_doc,end_doc):
    print(x)
    corpus=fileRead(x)
    documents.append(corpus)

data_words = list(sent_to_words(documents))


# See trigram example
#print(trigram_mod[bigram_mod[data_words[0]]])



# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# length greater than 2
data_words_nostops = length_check(data_words_nostops)



nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_lemmatized, min_count=5, threshold=30) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_lemmatized], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Form Bigrams
data_words_bigrams = make_bigrams(data_lemmatized)

# Form trigrams
#data_words_trigrams = make_trigrams(data_words_nostops)

data_final = data_words_bigrams
# Create Dictionary



id2word = corpora.Dictionary(data_final)



# Create Corpus
texts = data_final

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# =============================================================================
# #
# ## Build LDA model
#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=10, 
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)
# #Print the Keyword in the 10 topics
#pprint(lda_model.print_topics())
#doc_lda = lda_model[corpus]
# 
# =============================================================================
#______________________________________________________________________________
#
#
#mallet_path = 'C:/mallet/bin/mallet'
#ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=30, id2word=id2word, workers = 4,iterations = 100)
#optimal_model = ldamallet
#model_topics = optimal_model.show_topics(formatted=False)
#pprint(optimal_model.print_topics(num_words=10))
#
#df_topic_sents_keywords = format_dominant_topics(optimal_model, corpus, data_words)
#pprint(df_topic_sents_keywords.head(10))
#out_path = r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\4.2 Analyzing with Bigram and trigram (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Trial COdes\Model\dominant_topic_mallet_30_V2.csv'''
#df_topic_sents_keywords.to_csv(out_path, index=False)
#
#out_path = r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\4.2 Analyzing with Bigram and trigram (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Trial COdes\Model\topic_distribution_mallet_30_V2.csv'''
#df_topic_distribution=format_document_topic_distribution(optimal_model, corpus,30)
#df_topic_distribution.to_csv(out_path, index=False)
#print(df_topic_distribution)
#
#
##convert malletLda to lda
#ldamallet = malletmodel2ldamodel(ldamallet)
#print('prepare')
#pyLDAvis.enable_notebook()
#vis=pyLDAvis.gensim.prepare(ldamallet, corpus, id2word, mds='pcoa',sort_topics=False)
#pprint(ldamallet.show_topics(formatted=True))
##pyLDAvis.show(vis)
#pyLDAvis.save_html(vis, r'''D:\CLoud\Academic\Research\___\Photoshop COmmands\4.2 Analyzing with Bigram and trigram (Using 750 symmetrical data)\1. Feature (Words)\Topic Model\Trial COdes\pyLdavis\malletLDA_30topicsV2.html''')
#

#______________________________________________________________________________

#pprint(ldamallet[corpus[9]])


# Select the model and print the topics


# 
# =============================================================================
# 
# #Compute Perplexity
 # print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better
 # 
 # 
 # =============================================================================
 
 
#coherence_model_lda = CoherenceModel(model=ldamallet, texts=texts_2, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)


# =============================================================================
 # initialization of the values for plotting
 
_start=20
_limit=100
_step=5
 
# # =============================================================================
# # # Can take a long time to run.
# # model_list, coherence_values, perplexity_values = compute_coherence_value_gensim(dictionary=id2word, corpus=corpus, texts=data_final,  limit=_limit, start=_start, step=_step)
# # 
# # =============================================================================
# 
#Can take a long time to run.
model_list, coherence_values = compute_coherence_value_mallet(dictionary=id2word, corpus=corpus, texts=data_final,  limit=_limit, start=_start, step=_step)
 
# Show graph limit, start, step
x = plotCoherence(_limit,_start, _step,coherence_values)
 
# # =============================================================================
# # comment out if using gensim LDA model
# # y = plotPerplexity(_limit,_start, _step,perplexity_values)
# # =============================================================================
#  
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# 
# 
# =============================================================================
#Visualize the topics


# =============================================================================
# 
# fileWrite(61,str(data_words_bigrams[4]),len(data_words_bigrams[4]))
# print(data_words_bigrams[4])
# print(len(data_words_bigrams[4]))
# 
# fileWrite(61,str(data_words_nostops[4]),len(data_words_nostops[4]))
# print(data_words_nostops[4])
# print(len(data_words_nostops[4]))
# =============================================================================
