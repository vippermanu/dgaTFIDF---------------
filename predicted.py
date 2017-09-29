#encoding=utf-8
import sys
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score
import re
import time

if __name__ == '__main__':

   # vocabulary = ['word', 'hello', 'apple', 'milk']
   # tv = TfidfVectorizer(analyzer=u'word', vocabulary = vocabulary)
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
#为了解决测试集vocabulary维度不同，先加载训练集的vocabulary
    #参考文章http://blog.csdn.net/u013083549/article/details/51262721?locationNum=2
    pattern = re.compile(r'(\w|\d)*\.(com|net|cn|org|cc|gov|edu\.cn|cc|co|\w{1,10})$')
    tv = joblib.load('py27mix360.m')
    model = joblib.load('py27mix360.pkl')

    print('*************')
    fp = open('wubaoDNS.txt',  'r')
    #fp = open('test1.txt', 'w+')
    uris = fp.readlines()
    for uri in uris:
        uriPrint = uri.strip('\n')
        #print(uri)
        m = re.search(pattern, uri, flags=0)
        if m:
          #  print(m.group(0))
            uri = m.group(0)
        uri = re.sub("(\.com)|(\.cc)|(\.us)|(\.org)|(\.cn)|(\.\w{1,10})|(\.\w{1,10}\.\w{1,10})", "", uri)
        uri = uri.strip('\n')
      #  print(uri)
       # print(time.time())
        df = []
        df.append(uri)
        n_grams_tfidf = tv.transform(df)



        predicted = model.predict(n_grams_tfidf)
       # print(time.time())
        if (predicted):

           # print(uri)
            print('devil url')
            print('host = '+ uri)
            print('url = '+uriPrint)
            print('!!!!!!!!!!!!!!!!')
           # fp2.writelines(uriPrint)
        else:
           # print('safe url')
           # print('host = '+ uri)
           # print('url = '+uriPrint)
           # print('!!!!!!!!!!!!!!!!')
            pass

    '''
    print('start')
    for i in range(0,len(predicted)):
        if(predicted[i]!= df['label'][i] ):
            print((str(df['uri'][i]))+'预测的'+ str(predicted[i]) +'实际的'+str(df['label'][i] ))
    print('end')
    print("Classifier accuracy:", accuracy_score(df['label'], predicted))

'''




   # n_grams_train = count_vectorizer.fit_transform(x_train['uri'])
   # n_grams_dev = count_vectorizer.transform(x_dev['uri'])



   # n_grams_dev = count_vectorizer.transform(x_dev['uri'])
  #  print('Number of features:', len(count_v1.vocabulary_))