from collections import defaultdict
import pickle

from gluonnlp.data import SpacyTokenizer
import mxnet.ndarray as nd
import numpy as np
from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

fst = lambda x: x[0]
snd = lambda x: x[1]

tok2idx = pickle.load(open('tok2idx', 'rb'))
tokenizer = SpacyTokenizer()
def tokenize(x):
    d = defaultdict(lambda: 0)
    for tok in tokenizer(x.title):
        idx = tok2idx.get(tok, 0)  # TODO default value
        d[idx] += 1
    if d:
        n = sum(d.values())
        return [[x.nid, len(d)], [[k, d[k] / n] for k in sorted(d)]]
    else:
        return [[x.nid, 0], tuple()]

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
post_df = ss.read.json('top50-posts.txt').select('id', 'title', 'subreddit_id')
ids = post_df.select('id').rdd.flatMap(lambda x: x).collect()
nids = sc.parallelize(zip(ids, range(len(ids)))).toDF(['id', 'nid'])
post_df = post_df.join(nids, 'id')

'''
embeddings = nd.array(np.load('embeddings.npy'))
x_rdd = post_df.select('nid', 'title').rdd.map(tokenize).cache()
indptr  = nd.array(np.cumsum([0] + x_rdd.map(fst).map(snd).collect()))
indices = nd.array(x_rdd.map(snd).flatMap(lambda x: x).map(fst).collect())
data = nd.array(x_rdd.map(snd).flatMap(lambda x: x).map(snd).collect())
shape = [len(indptr) - 1, len(embeddings)]
matrix = nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
argsort_x = np.argsort(x_rdd.map(fst).map(fst).collect())
x = nd.sparse.dot(matrix, embeddings).asnumpy()[argsort_x]

y_rdd = post_df.select('nid', 'subreddit_id').rdd
argsort_y = np.argsort(y_rdd.map(lambda d: d['nid']).collect())
y = np.array(y_rdd.map(lambda d: d['subreddit_id']).collect())[argsort_y]
unique_y, inverse_y = np.unique(y, return_inverse=True)
y = np.arange(len(unique_y))[inverse_y]

pickle.dump(nids.select('id').flatMap(lambda x: x).collect(), open('ids.pickle', 'wb'))
np.save('x', x)
np.save('y', y)
'''

cmnt_df = ss.read.json('top50-comments.txt').select('author', 'link_id')
cmnt_df = cmnt_df.withColumn('id', regexp_replace('link_id', 't3_', '')).join(nids, 'id')
edges = cmnt_df.alias('u').join(cmnt_df.alias('v'), 'author').sample(fraction=0.1)
u = np.array(edges.select('u.nid').rdd.flatMap(lambda x: x).collect())
v = np.array(edges.select('v.nid').rdd.flatMap(lambda x: x).collect())
np.save('u', u)
np.save('v', v)
