import numpy as np
from pyspark.sql.functions import regexp_replace, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType

ss = SparkSession.builder.getOrCreate()
int36 = udf(lambda x: int(x, 36), IntegerType())
post_df = ss.read.json('top50-posts.txt').select('id', 'title', 'subreddit_id').withColumn('pid', int36('id'))
pid = post_df.select('id').rdd.flatMap(lambda x: x).collect()
post_df = post_df.join(sc.parallelize(zip(pid, range(len(pid)))).toDF(['pid', 'nid']), 'pid')
cmnt_df = ss.read.json('top50-comments.txt').select('author', 'link_id').withColumn('id', int36(regexp_replace('link_id', 't3_', '')))
