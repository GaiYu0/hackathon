import argparse
import pickle

import pandas as pd
from pyspark.sql.session import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument('--rs', type=str, nargs='+')
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
post_df = None
for f in args.rs:
    df = ss.read.json(f).select('id', 'subreddit', 'subreddit_id', 'title')
    post_df = df if post_df is None else post_df.union(df)
subreddit_ids = pickle.load(open('subreddit_ids', 'rb'))
ret = post_df.filter(post_df.subreddit_id.isin(*subreddit_ids)).coalesce(1)
ret.write.orc('RS.orc', mode='overwrite')
ret.write.json('RS.json', mode='overwrite')
