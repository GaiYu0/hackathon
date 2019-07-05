import argparse
import pandas as pd
from pyspark.sql.session import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument('--rs', type=str, nargs='+')
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
post_df = None
for f in args.rs:
    df = ss.read.json(f)
    post_df = df if post_df is None else post_df.union(df)
subreddits = pickle.load(open('subreddits', 'rb'))
post_df.isin(*subreddits).to_json('RS')
