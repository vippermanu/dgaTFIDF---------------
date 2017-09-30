#encoding=utf-8
import sys
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score
import re
import time
import numpy as np
#import gib_detect_train
import pickle
import scipy
import re
from scipy.sparse import csr_matrix
from scipy import sparse


def get_aeiou(domain_list):
  x = []
  y = []
  for domain in domain_list:
   # print(domain)
    x.append(len(domain))
    count = len(re.findall(r'[aeiou]',domain.lower()))
    count = (0.0 + count)/len(domain)
    y.append(count)
  return x,y


def get_uniq_char_num(domain_list):
  x = []
  y = []
  for domain in domain_list:
    x.append(len(domain))
    count = len(set(domain))
    count = (0.0+count)/len(domain)
    y.append(count)
  return y

if __name__ == '__main__':

   # vocabulary = ['word', 'hello', 'apple', 'milk']
   # tv = TfidfVectorizer(analyzer=u'word', vocabulary = vocabulary)
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
#为了解决测试集vocabulary维度不同，先加载训练集的vocabulary
    #参考文章http://blog.csdn.net/u013083549/article/details/51262721?locationNum=2
    pattern = re.compile(r'(\w|\d)*\.(com|net|cn|org|cc|gov|edu\.cn|cc|co|\w{1,10})$')
    tv = joblib.load('total.m')
    model = joblib.load('total.pkl')

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

        urisCount, AEIOUs = get_aeiou(df)
        AEIOUs = np.array(AEIOUs)
        urisCount = len(urisCount)
        AEIOUs = AEIOUs.reshape(urisCount, 1)
        AEIOUs = sparse.csr_matrix(AEIOUs)

   # 拼接不重复字符特征到最后一列
        uniq_char_num = get_uniq_char_num(df)
        uniq_char_num = np.array(uniq_char_num)
        uniq_char_num = uniq_char_num.reshape(urisCount, 1)
        uniq_char_num = sparse.csr_matrix(uniq_char_num)

# total = np.array(0)
        total = scipy.sparse.hstack((n_grams_tfidf, AEIOUs, uniq_char_num), format='csr')

        predicted = model.predict(total)
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