import argparse
import pickle

import pandas as pd
from pyspark.sql.session import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument('--rs', type=str, nargs='+')
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
cmnt_df = None
for f in args.rc:
    df = ss.read.json(f).select('author', 'created_utc', 'link_id', 'subreddit', 'subreddit_id')
    cmnt_df = df if cmnt_df is None else cmnt_df.union(df)
subreddit_ids = pickle.load(open('subreddit_ids', 'rb'))
cmnt_df.filter(cmnt_df.subreddit_id.isin(*subreddit_ids)).coalesce(1).write.json('RC', mode='overwrite')
