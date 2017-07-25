import re
import nltk
from os import listdir
from os.path import isfile, join
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = SnowballStemmer("english")

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

files = [f for f in listdir("/Users/sanket/Desktop/nlp_emailrecs/sample_data") if isfile(join("/Users/sanket/Desktop/nlp_emailrecs/sample_data", f)) and f.endswith(".email")]

all_emails = []
for file in files:
	f = open(join("/Users/sanket/Desktop/nlp_emailrecs/sample_data", file))
	text = f.read()
	f.close()
	all_emails.append(text)

alltokens = set()
for file in files[:1000]:
	f = open(join("/Users/sanket/Desktop/nlp_emailrecs/sample_data", file))
	text = f.read()
	f.close()
	alltokens = alltokens.union(tokenize_stop_stem(text))

print len(alltokens)

from sklearn.feature_extraction.text import TfidfVectorizer
#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_features=200000, min_df=0.2, stop_words='english', use_idf=True, tokenizer=tokenize_stop_stem)
tfidf_matrix = tfidf_vectorizer.fit_transform(all_emails)

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
%time km.fit(tfidf_matrix)
clusters = km.labels_.tolist()



#f = open('/Users/sanket/Desktop/nlp_emailrecs/sample_data/1a88724b17792136.email')
#text = f.read()