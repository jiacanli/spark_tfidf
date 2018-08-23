from __future__ import print_function
from pyspark import SparkContext
import sys
import math
from functools import reduce
from collections import Counter
from operator import itemgetter, attrgetter
from time import time
t=time() # PROGRAMM START TIME


#
#THIS PROGRAMME AIMS TO CACULATE THE TFIDF VALUE OF THE AIRTICLE
#BY THE WAY,CACULATE THE BIGGEST FILE LIKE WIKIPDIA WILL TAKE PROBABLY 45 MINUTES IN TOTAL ^_^

sc = SparkContext(appName="Pythoncount")

count=0

#FUNCTION INDEX()
#THE PARAMETER X SHOULD BE AN LIST
#THIS FUNCTION IS TO ADD INDEX TO EACH ELEMENT IN THE RDD

def index(x):
 list=[]
 for i in x[0]:
  list.append((i[0],(x[1],i[1])))
 return list


#FUNCTION CALU(X)
#THE PARAMETER X IS EXPECTED TO BE DICT TYPE
#THIS FUNCTION IS TO CACULATE THE TFIDF OF TERM(WORD)

def calu(x):
 global words_news_df_dict
 for i in x.keys():
  x[i]=x[i]*words_news_df_dict[i]
 return x


#STEP 1: READ ORIGIN RESOURCE FROM A LOCAL FILE WIKIPEDIA WHICH WILL TAKE 45 MIN TO CACULATE 

line1=sc.textFile('/home/ubuntu/Data/wikipedia') 
words_news=line1.map(lambda x:x.split(" "))
D=words_news.count()


#STEP 2: CACULATE THE TF CACULATE THE TF VALUE OF EACH TERM

words_news_tf=words_news.map(lambda x:Counter(x)).map(lambda y:dict(y)) ####(WORD,TF) splited type:dict
#print(words_news_tf.collect())

#STEP 3:CACULATE THE DF AND IDF VALUE OF EACH TERM 

words_news_df=words_news_tf.flatMap(lambda x:x).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y) #### (WORD,IDF)  type: dict
words_news_idf=words_news_df.map(lambda x:(x[0],math.log((D+1)/(x[1]+1))))

#TRANLATE THE WORDS_NEWS_DF WHICH TYPE IS RDD TO THE DICT TYPE
words_news_df_dict=dict(map(lambda x:x,words_news_df.collect()))   #### (WORD,IDF)  type: dict

#STEP 4:CACULATE THE FINAL TFIDF OF EACH TERM 
words_news_tfidf=words_news_tf.map(lambda x:calu(x))

#STEP 5:THE MAIN WORK OF THIS STEP IS ADD THE INDEX TO EVERY ELEMENTS IN RDD AND SORT THEM BY TFIDF VALUE AND SPECIFY THE FORMAT 
words_news_tfidf_index=words_news_tfidf.zipWithIndex().map(lambda x:(list(zip(x[0].keys(),x[0].values())),x[1])).map(lambda x:index(x)) ####ZIP RDD WITH INDEX BY USING ZIPWITHINDEX FUNCTION --------> TRASLATE THE FORMAT TO :(WORD,(ID,TFIDF)) 

#THE FOLLOWING CODE IS SOMETHING LIKE THIS :(WORD,(ID,TFIDF))-->SORT BY TFIDF-->(WORD,ID)--->REDUCE BY KEY--->(WORD,[ID...])
words_news_tfidf_sort=words_news_tfidf_index.flatMap(lambda x:x).sortBy(lambda x:x[1][1],ascending=True).map(lambda x:(x[0],x[1][0])).reduceByKey(lambda x,y:str(x)+","+str(y)).map(lambda x:(x[0],[x[1]]))

# SAVE THE RESULT TO THE LOCAL FILE
words_news_tf.saveAsTextFile('/home/ubuntu/Data/Result/Step2')
words_news_df.saveAsTextFile('/home/ubuntu/Data/Result/Step3')
words_news_tfidf.saveAsTextFile('/home/ubuntu/Data/Result/Step4')
words_news_tfidf_sort.saveAsTextFile('/home/ubuntu/Data/Result/Step5')


print("total running time:")
print(time()-t)






