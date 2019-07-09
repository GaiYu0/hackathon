import numpy as np
from pyspark.sql.functions import regexp_replace, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType

ss = SparkSession.builder.getOrCreate()
int36 = udf(lambda x: int(x, 36), IntegerType())
cmnt_df = ss.read.json('top50-comments.txt').select('author', 'link_id').withColumn('link_id', int36(regexp_replace('link_id', 't3_', '')))
edges = cmnt_df.alias('u').join(cmnt_df.alias('v'), 'author')
u = np.array(edges.select('u.link_id').collect())
v = np.array(edges.select('v.link_id').collect())
unique_u, inverse_u = np.unique(u, return_inverse=True)
nodes = np.arange(len(unique))
src = nodes[inverse_u]
unique_v, inverse_v = np.unique(v, return_inverse=True)
dst = nodes[inverse_v]
assert np.all(unique_u == unique_v)
