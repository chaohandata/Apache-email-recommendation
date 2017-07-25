
# coding: utf-8

# # Email recommendations using cosine similarity on Apache-Drill mailing lists
# 
# ### Execution time:
# ##### Total execution time is less than 2 mins.
#     1. Generating TF-IDF matrix
#         CPU times: user 1min 6s, sys: 168 ms, total: 1min 7s
#         Wall time: 1min 7s
#     2. Calculating cosine similarities
#         CPU times: user 22.9 s, sys: 10.2 s, total: 33.1 s
#         Wall time: 35.9 s
#     
# ### Memory usage:
# ##### Total memory required approx 4GB
#     1. TF-IDF Sparse Matrix:
#         1MB
#     2. Pairwise Cosine-similarity matrix:
#         4025MB ~ 3.93GB
# 
# ### Result quality:
#    Results look promising, top 10 recommendations for each email are stored in a csv file. In the testing portion of the script you can specify any email and get the recommendations. Just eyeball it to see how close they are.
#     
# ### Issues:
#    **The size of the cosine_similarity matrix is the problem.** It will be too large as the size of the data grows.
#    Next steps will be to use Locality Sensitive Hashing and get 15 approximate nearest neighbours and then re-compute cosine similarities on the set of 15 neighbours and then store the top 10 similar emails
# 

# In[3]:


import re
import nltk
import pandas as pd
import numpy as np
from sys import getsizeof
from os import listdir
from os.path import isfile, join
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[5]:


tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = SnowballStemmer("english")


# In[6]:


# tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
def tokenize_stop_stem(text):
    tokens = tokenizer.tokenize(text)
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation) and stem
    filtered_tokens = set()
    for token in tokens:
        token = token.lower()
        if token not in stopwords:
            if not re.search('[0-9]', token):
                try:
                    token = stemmer.stem(token)
                    filtered_tokens.add(token)
                except UnicodeDecodeError:
                    print 'illeagal token ignored:',token
                    pass
    return filtered_tokens


# In[7]:


files = [f for f in listdir("/Users/sanket/Desktop/nlp_emailrecs/sample_data") if isfile(join("/Users/sanket/Desktop/nlp_emailrecs/sample_data", f)) and f.endswith(".email")]


# In[8]:


getsizeof(files)


# In[9]:


all_emails = []
for file in files:
    f = open(join("/Users/sanket/Desktop/nlp_emailrecs/sample_data", file))
    text = f.read()
    f.close()
    all_emails.append(text)


# In[28]:


print sum([getsizeof(k) for k in all_emails])/10**6,'MB'


# In[22]:


# alltokens = set()
# for file in files:
#    f = open(join("/Users/sanket/Desktop/nlp_emailrecs/sample_data", file))
#    text = f.read()
#    f.close()
#    alltokens = alltokens.union(tokenize_stop_stem(text))
# print len(alltokens)


# ### Generate TF-IDF matrix on the emails we have

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[12]:


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_features=2000000, stop_words='english', use_idf=True, tokenizer=tokenize_stop_stem)


# In[13]:


get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(all_emails)')
# terms = tfidf_vectorizer.get_feature_names()


# In[27]:


print sum([getsizeof(k) for k in tfidf_matrix])/10**6,'MB'
print 'shape:',tfidf_matrix.shape


# ### Generate pairwise cosine similartiy
# Distance = 1 - similarity

# In[17]:


from sklearn.metrics.pairwise import cosine_similarity


# In[18]:


get_ipython().magic(u'time cosine_sim = cosine_similarity(tfidf_matrix)')


# In[20]:


getsizeof(cosine_sim)/10**6


# In[14]:


cosine_sim.mean()


# In[30]:


x=cosine_sim[0]
x.argsort()[::-1][:10]


# ### Top 10 similar emails based on Cosine similarity saved to similarity_results.csv

# In[51]:


ls=[]
for i in range(cosine_sim.shape[0]):
    #print i
    temp = []
    temp.append(files[i])
    x = cosine_sim[i].argsort()[::-1][1:11]
    for j in x:
        temp.append(files[int(j)])
    ls.append(temp)

dataFrame = pd.DataFrame(ls)
dataFrame.head()
dataFrame.to_csv('/Users/sanket/Desktop/nlp_emailrecs/similarity_results.csv',index=False)


# # Test recommendations

# In[1]:


import pandas as pd


# In[2]:


recommendations = pd.read_csv('/Users/sanket/Desktop/nlp_emailrecs/similarity_results.csv')
recommendations.head()


# In[12]:


recommendations.shape


# In[31]:


index = np.random.randint(0,22430)
index


# In[32]:


mainEmail = recommendations.iloc[index,:][0]
mainEmail = open('/Users/sanket/Desktop/nlp_emailrecs/sample_data/'+mainEmail).read()
print mainEmail


# In[34]:


recEmail1 = recommendations.iloc[index,:][5]
recEmail1 = open('/Users/sanket/Desktop/nlp_emailrecs/sample_data/'+recEmail1).read()
print recEmail1


# ### Kmeans clustering of TF-IDF vectors

# In[18]:


from sklearn.cluster import KMeans


# In[19]:


def cluster_gridsearch(num_clusters):
    km = KMeans(n_clusters=num_clusters,n_jobs=-1)
    get_ipython().magic(u'time km.fit(tfidf_matrix)')
    print km.inertia_
    return km.inertia_


# In[33]:


num_clusters = 5
km = KMeans(n_clusters=num_clusters,n_jobs=-1)


# In[34]:


get_ipython().magic(u'time km.fit(tfidf_matrix)')


# In[52]:


label_df = pd.DataFrame(km.labels_)
label_df[0].value_counts()


# In[59]:


km.inertia_


# In[20]:


error_list = []


# In[21]:


for i in range(5,15,5):
    print i,
    error_list.append(cluster_gridsearch(i))


# In[22]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[23]:


plt.plot(error_list)
plt.show()


# In[25]:


plt.plot(error_list)
plt.show()


# ### Heirarchical Clustering on the data 

# In[39]:


# Warnning it will take too long to run. Remove comments to execute
from scipy.cluster.hierarchy import ward, dendrogram
dist = 1 - cosine_sim
# linkage_matrix = ward(dist)
# fig, ax = plt.subplots(figsize=(15, 20))

