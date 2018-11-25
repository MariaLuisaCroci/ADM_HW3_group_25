import csv
import pandas as pd
import numpy as np
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import string
from nltk.tokenize import RegexpTokenizer
#download the stop words
nltk.download("stopwords")
import json
import math


def clean(doc):
    tokenizer = RegexpTokenizer("[\w'\$]+") #we don't want to split the dollar $ from his price
    doc = tokenizer.tokenize(doc) #tokenize the query for stop word and punctuation
    doc = [t for t in doc if t not in stopwords] #remove stopword
    doc = [st.stem(word) for word in doc] #stemming
    
    return doc

#define a function for tokenizer, stemming, remove stop words and punctuation

stopwords = stopwords.words('english') # get english stop words

#we tokenize except for the symbol $ that remains with the value
tokenizer = RegexpTokenizer("[\w'\$]+")

#stemming
st = PorterStemmer()

#########searchEngine_1

#we create a function for the Search Engine 1

def make_clickable(val): #we will use this to make the urls in output clickable
        # target _blank to open new window
        return '<a target="_blank" href="{}">{}</a>'.format(val, val)

vocabulary = json.loads(open("vocabulary.json").read())
vocabulary = {k:v for v,k in vocabulary.items()} #change key as value and viceversa 

def searchEngine_1():
    query = input("Insert your query: ") #user input
    query = clean(query) #tokenizing, stemming, removing stop words and punctuation
    
    inverted_index = json.loads(open("inverted_index.json").read())


    for i in set(query):
        if i in vocabulary:
            query.remove(i)
            query.append(vocabulary[i])
        else:
            query.remove(i)

    docs = []
    query = set(query)
    try:
        for i in query:
            docs.append(set(inverted_index[i]))

    #try:
        docs = set.intersection(*docs)

    except: #make a print of exception "result for your query doesn't exist"

        return print("There are no results that match your query.") #it stops the function if there aren't match 

    results = []
    for file in docs:
        opentsv = open("data/doc_tsv/" + file + ".tsv", "r", encoding="utf8")
        lines = opentsv.readlines()
        for x in lines:
            results.append([x.split('\t')[7], x.split('\t')[4], x.split('\t')[2], x.split('\t')[8]])
        opentsv.close()

    result = pd.DataFrame(results, columns = ['Title', 'Description', 'City', 'URL'])
    result.Title = result.Title.str.replace('$', '$\$$') #escape the problem with latex codification
    result.Description = result.Description.str.replace('$', '$\$$') #escape the problem with latex codification

    result.style.set_properties(*{'text-align': 'center'})
    
    return result.style.format({'URL': make_clickable}) #make_clickable function




#########searchEngine_2

corpus = json.loads(open('index.json').read())
inverted_index = json.loads(open('inverted_index.json').read())
vocab = json.loads(open('vocabulary.json').read())
idf = json.loads(open('tfidf_index.json').read())

idf = {}
for term in inverted_index:
    term1 = vocab[term]
    idf[term] = math.log(1+(len(corpus)/len(term1)))
for term in inverted_index:
    term1 = vocab[term]
    for doc in inverted_index[term]:
        count = (corpus[doc].count(term1)/len(corpus[doc]))*idf[term]
        inverted_index[term][inverted_index[term].index(doc)]=(doc,count)


def searchEngine_2():
    query = input("Insert your query: ")
    k = int(input("How many results do you want to see? "))
    error = []

    query = clean(query)


    for i in query:
        if i in vocabulary:
            query[query.index(i)] = vocabulary[i] #replacing word with id in the query
        else:
            error.append(i) #if is not in vocabulary we append in error
            #query.remove(i) #and remove it from query

    if error != []: #if error not empty
        return print('there are no results that match your search criteria: '+', '.join(error))
    else: #else do the search engine 
        query_tfidf = {}

        for i in query:
            query_tfidf[i] = (query.count(i)/len(query))*(idf[i]) #calculate tf-idf of the words in query

        result_docs = []

        for i in query:
            docs = []
            for j in inverted_index[i]: #access to the number of the query in inverted_index
                docs.append(j[0])  #append the doc without tfidf in docs
            result_docs.append(set(docs)) # distinct documents

        result_docs = list(set.intersection(*result_docs)) #intersection of the documents 

        docsidf = {} 
        for i in query:
            #taking the whole tuple and not single doc
            docsidf[i] = [list(item) for item in inverted_index[i] if item[0] in result_docs]
            
        #product = 0
        docs_norm = {}
        dotproducts = {}

        for name in result_docs: #for each doc in result_docs
            product = 0
            dotproducts[name] = 0
            docs_norm[name] = 0
            for term in query: #for each word in query
                #dotproducts[name] = 0
                # it take the tf-idf score from docsidf at key equal term for the document name 
                idf1 = [x[1] for x in docsidf[term] if x[0] == name]
                product += idf1[0] 
                docs_norm[name] += product**2    
                #make the product of the tfidf between query and doc, and sum the products for each word
                dotproducts[name] += query_tfidf[term]*idf1[0] 
            docs_norm[name] = math.sqrt(docs_norm[name]) 

        query_norm = 0
        for x in query_tfidf:
            query_norm += query_tfidf[x]
        dot_product = 0

        cosine_similarity = {}

        for product in dotproducts:
            cosine_similarity[product]=(dotproducts[product]/(docs_norm[product]*query_norm))
        cosine_sorted=[]

        for key, value in sorted(cosine_similarity.items(), key=lambda item: (item[1],item[0]),reverse=True):
            cosine_sorted.append(key)
        cosine_sorted=cosine_sorted[:k]

        results=[]

        for file in cosine_sorted:
            opentsv=open("data/doc_tsv/"+file+".tsv","r",encoding="utf8")
            lines=opentsv.readlines()

            for x in lines:
                results.append([x.split('\t')[7],x.split('\t')[4],x.split('\t')[2],x.split('\t')[8],cosine_similarity[file]])
            opentsv.close()

        result=pd.DataFrame(results,columns=['Title','Description','City','URL','Similarity'])
        result.Title = result.Title.str.replace('$', '$\$$') #escape the problem with latex codification
        result.Description = result.Description.str.replace('$', '$\$$') #escape the problem with latex codification

    result.style.set_properties(*{'text-align': 'center'})
    
    return result.style.format({'URL': make_clickable})


########Search Engine 3


def city_score(df):
    if df['price'] < df['min'] + df['min_int']:
        return 5
    elif df['price'] < df['min'] + 2*df['min_int']:
        return 4 
    elif df['price'] < df['mean'] + df['max_int']:
        return 3
    elif df['price'] < df['mean'] + 2*df['max_int']:
        return 2
    else:
        return 1

scores_dict = json.loads(open('scores_dict.json').read())


def searchEngine_3():
    query = input("Insert your query: ") #user input
    
    filters = list(input("please enter the filters you want separated by space or leave blank:\ntype\n1 for city\n2 for bedrooms\n3 for price\n").split())
                
    

    query = clean(query) #tokenizing, stemming, removing stop words and punctuation
    
    inverted_index = json.loads(open("inverted_index.json").read())


    for i in set(query):
        if i in vocabulary:
            query.remove(i)
            query.append(vocabulary[i])
        else:
            query.remove(i)
    
    docs = []
    query = set(query)
    
    try:
        for i in query:
            docs.append(set(inverted_index[i]))

    
        docs = set.intersection(*docs)

    except: #make a print of exception "result for your query doesn't exist"

        return print("Result for your query doesn't exist.") #it stops the function if there aren't match 

    results = []
    
    
    
    for file in docs:
        opentsv = open("data/doc_tsv/" + file + ".tsv", "r", encoding="utf8")
        lines = opentsv.readlines()
        for x in lines:
            results.append([x.split('\t')[7], x.split('\t')[4], x.split('\t')[2],
                            int(x.split("\t")[0].replace("$", "")), int(x.split("\t")[1].replace("Studio", "1")),
                            x.split('\t')[8], scores_dict[file]])
        opentsv.close()
    

    result = pd.DataFrame(results, columns = ['Title', 'Description', 'City', "Price", "Bedroom", 'URL', "Score"])
    result.Title = result.Title.str.replace('$', '$\$$') #escape the problem with latex codification
    result.Description = result.Description.str.replace('$', '$\$$') #escape the problem with latex codification
    
    result = result.sort_values(by=['Score'], ascending=False)
    result = result.reset_index(drop=True)
    result.index += 1
    
    result.style.set_properties(*{'text-align': 'center'})
    
    if filters != []:
        #filters = map(int,filters)
        if "1" in filters :
            city = input("Type the name of a city: ")
            result = result[result.City==city]
            
        if "2" in filters :
            min_bed,max_bed = list(map(int,input("Type the minimun and maximum number of bedroom separeted by space: ").split()))
            result = result[result.Bedroom <= max_bed]
            result = result[result.Bedroom >= min_bed ]
        if "3" in filters :
            min_price, max_price = list(map(int,input("Type the minimun and maximum price separeted by space: ").split()))
            result = result[result.Price <= max_price ]
            result = result[result.Price >= min_price ]
        result = result.reset_index(drop=True)
        result.index += 1
    idx=0
    result.insert(idx, "Ranking", result.index)
    
    result = result.drop(["Price", "Bedroom", "Score"],axis=1)
    
    return result.style.format({'URL': make_clickable}) #make_clickable function
   

