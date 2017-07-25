from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
# cosine_sim is a nDoc*nDoc similarity matrix I choose the to 10
x=cosine_sim[0]
x.argsort()[::-1][:10]