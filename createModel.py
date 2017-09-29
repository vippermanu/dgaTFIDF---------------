#encoding=utf-8
import importlib,sys
#importlib.reload(sys)
import sys
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
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
    print(domain)
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
  #  model_data = pickle.load(open('gib_model.pki', 'rb'))

   # df = pd.read_csv('dgaNOcom.csv',encoding='utf-8')
  #  df1 = pd.read_csv('dga.csv', encoding='utf-8')
  #  print(df1['label'][0])
    df1 = pd.read_csv('dnsW.csv', encoding='utf-8')
    df2 = pd.read_csv('dnsB.csv', encoding='utf-8')
    frames = [df1, df2]
    df = pd.concat(frames,keys=['uri','label'])

   # print(df['uri'][0])
   # print(df['uri'][1520000])
    print(len(df['uri']))
    print(df['uri'][1])
    print(df['uri'][15])
    print('########################')



  #  print(df[df['label'] == 1].head())
    print('????????????????????????')
    attributes = ['uri', 'label']
   # x_train, x_test, y_train, y_test = train_test_split(df[attributes], df['label'], test_size=0.1,
   #                                                     stratify=df['label'], random_state=0)



   # x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1,
    #                                                  stratify=y_train, random_state=0)


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

    # 拼接不重复字符特征到最后一列
    uniq_char_num = get_uniq_char_num(df['uri'])
    uniq_char_num = np.array(uniq_char_num)
    uniq_char_num = uniq_char_num.reshape(urisCount,1)
    uniq_char_num = sparse.csr_matrix(uniq_char_num)


    #total = np.array(0)
    total = scipy.sparse.hstack((n_grams_tfidf, AEIOUs, uniq_char_num), format='csr')





    print(type(total[0]))
    print((total[0]))

    from sklearn.tree import DecisionTreeClassifier

    print('start make model')
    #clf = DecisionTreeClassifier(random_state=0).fit(n_grams_tfidf, df['label'])
    clf = DecisionTreeClassifier(random_state=0).fit(total, df['label'])


    from sklearn.externals import joblib
    joblib.dump(clf,'total.pkl')

    joblib.dump(tv, "total.m")
  #  x = '12345'

  #  print(clf.predict(x))



   # n_grams_train = count_vectorizer.fit_transform(x_train['uri'])
   # n_grams_dev = count_vectorizer.transform(x_dev['uri'])



   # n_grams_dev = count_vectorizer.transform(x_dev['uri'])
  #  print('Number of features:', len(count_vectorizer.vocabulary_))