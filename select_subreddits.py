import argparse

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType

parser = argparse.ArgumentParser()
parser.add_argument('--rc', type=str, nargs='+')

args = parser.parse_args()

cmnt_df = None
for f in args.rc:
    df = ss.read.json(f).select()
    cmnt_df = df if cmnt_df is None else cmnt_df.union(df)
cmnt_df.groupBy('subreddit_id').count().toPandas().sort_values(by='count', ascending=False).reset_index().to_hdf('count', '/df')
