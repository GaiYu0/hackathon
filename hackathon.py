import numpy as np
from pyspark.sql.functions import regexp_replace, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType

ss = SparkSession.builder.getOrCreate()
int36 = udf(lambda x: int(x, 36), IntegerType())
cmnt_df = ss.read.json('top50-comments.txt').select('author', 'link_id').withColumn('link_id', int36(regexp_replace('link_id', 't3_', '')))
edges = cmnt_df.alias('src').join(cmnt_df.alias('dst'), 'author')
src = np.array(edges.select('src.link_id').collect())
dst = np.array(edges.select('dst.link_id').collect())
