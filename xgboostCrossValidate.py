#encoding=utf-8
import importlib,sys
import sys
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#import gib_detect_train
import pickle
import scipy
import pickle
import scipy
import re
from scipy.sparse import csr_matrix
from scipy import sparse
from collections import Counter
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

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

    print(len(df['uri']))
    print(df['uri'][1])
    print(df['uri'][150])
    print('########################')

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

    tv = TfidfVectorizer(analyzer='char', ngram_range={2, 3})
    n_grams_tfidf = tv.fit_transform(df['uri'].values.astype('U'))
    # 稀疏矩阵转为narray
    # n_grams_tfidf = n_grams_tfidf.toarray()
    # 拼接AEIOU特征到最后一列
    urisCount, AEIOUs = get_aeiou(df['uri'])
    AEIOUs = np.array(AEIOUs)
    urisCount = len(urisCount)
    AEIOUs = AEIOUs.reshape(urisCount, 1)
    AEIOUs = sparse.csr_matrix(AEIOUs)


    # 拼接不重复字符特征到最后一列
#    uniq_char_num = get_uniq_char_num(df['uri'])
#    uniq_char_num = np.array(uniq_char_num)
#    uniq_char_num = uniq_char_num.reshape(urisCount, 1)
#    uniq_char_num = sparse.csr_matrix(uniq_char_num)


#香浓熵作为特征
    Feashanon = calShanon(df['uri'])
    Feashanon = np.array(Feashanon)
    Feashanon = Feashanon.reshape(urisCount, 1)
    Feashanon = sparse.csr_matrix(Feashanon)

    # 域名长度作为特征
   # Fealen = get_domain_legnth(df['uri'])
   # Fealen = np.array(Fealen)
  #  Fealen = Fealen.reshape(urisCount, 1)
  #  Fealen = sparse.csr_matrix(Fealen)

    total = scipy.sparse.hstack((n_grams_tfidf, AEIOUs, Feashanon), format='csr')
   # pca = PCA(n_components=0.95)
   # pca_total = pca.fit_transform(total)
    print('????????????????????????')
    attributes = ['uri', 'label']
    x_train, x_test, y_train, y_test = train_test_split(total, df['label'], test_size=0.5,
                                                        stratify=df['label'], random_state=0)


    from sklearn.tree import DecisionTreeClassifier

   # clf = RandomForestClassifier().fit(x_train, y_train)
    #clf = ensemble.AdaBoostClassifier(learning_rate=1, n_estimators=500).fit(x_train, y_train)
   # clf = DecisionTreeClassifier(random_state=0).fit(x_train, y_train)
    #from sklearn.svm import SVC
    #clf = SVC().fit(x_train, y_train)
    clf = XGBClassifier()
    clf.fit(x_train, y_train)
    plot_importance(clf,max_num_features=50)
    pyplot.show()
    from sklearn.metrics import accuracy_score

    test_predicted = clf.predict(x_test)
    #print("DeciTreeClassifier accuracy:", accuracy_score(y_test, test_predicted))
    print("AdaBoostClassifier accuracy:", accuracy_score(y_test, test_predicted))

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    model = XGBClassifier()
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(total, df['label'])


   # eval_set = [(x_test, y_test)]
    #clf.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

    from sklearn.externals import joblib
  #  joblib.dump(clf,'mix360.pkl')

  #  joblib.dump(tv, "mix360.m")
  #  x = '12345'

  #  print(clf.predict(x))



   # n_grams_train = count_vectorizer.fit_transform(x_train['uri'])
   # n_grams_dev = count_vectorizer.transform(x_dev['uri'])



   # n_grams_dev = count_vectorizer.transform(x_dev['uri'])
  #  print('Number of features:', len(count_vectorizer.vocabulary_))