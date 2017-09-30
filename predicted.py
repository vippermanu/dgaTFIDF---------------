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
from collections import Counter
import math
from sklearn.ensemble import RandomForestClassifier

def get_domain_legnth(domain_list):
  y = []
  for domain in domain_list:
    lenDomain = len(domain)
    y.append(lenDomain)
  return y

def calShanon(domain_list):
  y = []
  for domain in domain_list:
    lenDomain = len(domain)
    f_len = float(lenDomain)
    count = Counter(i for i in domain).most_common()
    entropy = -sum(j/f_len * (math.log(j / f_len)) for i, j in count)  # shannon entropy
    y.append(entropy)
  return y

def get_aeiou(domain_list):
  x = []
  y = []
  for domain in domain_list:
   # print(domain)
    lenDomain = len(domain)
    x.append(lenDomain)
    count = len(re.findall(r'[aeiou]',domain.lower()))
    count = (0.0 + count)/lenDomain
    y.append(count)
  return x,y


def get_uniq_char_num(domain_list):
  y = []
  for domain in domain_list:
    lenDomain = len(domain)
    count = len(set(domain))
    count = (0.0+count)/lenDomain
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
    #fp = open('test1.txt', 'r')
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
        #    uniq_char_num = get_uniq_char_num(df)
        #    uniq_char_num = np.array(uniq_char_num)
        #    uniq_char_num = uniq_char_num.reshape(urisCount, 1)
        #    uniq_char_num = sparse.csr_matrix(uniq_char_num)


        # 香浓熵作为特征
        Feashanon = calShanon(df)
        Feashanon = np.array(Feashanon)
        Feashanon = Feashanon.reshape(urisCount, 1)
        Feashanon = sparse.csr_matrix(Feashanon)

        # 域名长度作为特征
 #       Fealen = get_domain_legnth(df)
   #     Fealen = np.array(Fealen)
   #     Fealen = Fealen.reshape(urisCount, 1)
   #     Fealen = sparse.csr_matrix(Fealen)

        total = scipy.sparse.hstack((n_grams_tfidf, AEIOUs, Feashanon), format='csr')

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