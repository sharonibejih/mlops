#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip freeze | grep scikit-learn')

import pickle
import pandas as pd

import os

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# Load March data
df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-03.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# mean predicted duration
print(y_pred.mean())

# year = 2021
# month = 2

# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# pd.Series(y_pred, name="predictions")


# df_result = pd.merge(df["ride_id"], pd.Series(y_pred, name="predictions"), 
# right_index=True, left_index=True)

# df_result.to_parquet(
#     "result.parquet",
#     engine="pyarrow",
#     compression=None,
#     index=False
# )

# df_result

# # get file size
# os.path.getsize("result.parquet")/1000000


