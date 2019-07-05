import argparse
import pickle

import pandas as pd
from pyspark.sql.session import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument('--rc', type=str, nargs='+')
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
post_df = ss.read.orc('RS.orc')
cmnt_df = None
for f in args.rc:
    df = ss.read.json(f).select('author', 'created_utc', 'link_id').withColumnRenamed('link_id', 'id')
    cmnt_df = df if cmnt_df is None else cmnt_df.union(df)
cmnt_df.join(post_df.select('id'), 'id').coalesce(1).write.json('RC.json', mode='overwrite')
