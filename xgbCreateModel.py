#encoding=utf-8
import importlib,sys
#importlib.reload(sys)
import sys
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import scipy
import re
from scipy.sparse import csr_matrix
from scipy import sparse
from collections import Counter
import math
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost as xgb


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
    df1 = pd.read_csv('dnsW.csv', encoding='utf-8')
    df2 = pd.read_csv('dnsB.csv', encoding='utf-8')
    frames = [df1, df2]
    df = pd.concat(frames,keys=['uri','label'])


    print('########################')



    print('????????????????????????')
    attributes = ['uri', 'label']



    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer

    tv = TfidfVectorizer(analyzer='char',ngram_range={2,3})
    n_grams_tfidf = tv.fit_transform(df['uri'].values.astype('U'))
    #稀疏矩阵转为narray
   # n_grams_tfidf = n_grams_tfidf.toarray()
    # 拼接AEIOU特征到最后一列
    urisCount, AEIOUs = get_aeiou(df['uri'])
    AEIOUs = np.array(AEIOUs)
    urisCount = len(urisCount)
    AEIOUs = AEIOUs.reshape(urisCount,1)
    AEIOUs = sparse.csr_matrix(AEIOUs)


      # 香浓熵作为特征
    Feashanon = calShanon(df['uri'])
    Feashanon = np.array(Feashanon)
    Feashanon = Feashanon.reshape(urisCount, 1)
    Feashanon = sparse.csr_matrix(Feashanon)


    total = scipy.sparse.hstack((n_grams_tfidf, AEIOUs, Feashanon), format='csr')


    print('start make model')

    clf = XGBClassifier(learning_rate=1)
    clf.fit(total, df['label'])



    from sklearn.externals import joblib

    joblib.dump(clf,'xgbtotal.pkl')

    joblib.dump(tv, "xgbtotal.m")
