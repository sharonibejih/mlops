import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):

    date_format = "%Y-%m-%d"

    if date==None:
        given_date = str(date.today())

    else:
        given_date = str(date)

    dtObj = datetime.strptime(given_date, date_format)

    train_n, valid_n = 2, 1
    # subtract n months from given date
    train_data_date = (dtObj - relativedelta(months=train_n)).date()
    train_data_date = (train_data_date.strftime(date_format))[:-3] # select YY and MM

    valid_data_date = (dtObj - relativedelta(months=valid_n)).date()
    valid_data_date = (valid_data_date.strftime(date_format))[:-3]

    train_path = f'../week1/data/fhv_tripdata_{train_data_date}.parquet' 
    val_path = f'../week1/data/fhv_tripdata_{valid_data_date}.parquet'

    return train_path, val_path

@flow
def main(date="2021-08-15"):

    # train_path: str = '../week1/data/fhv_tripdata_2021-01.parquet', 
    # val_path: str = '../week1/data/fhv_tripdata_2021-02.parquet'

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f"./models/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)
        f_out.close()

    with open(f"./models/dv-{date}.b", "wb") as f_out:
        pickle.dump(dv, f_out)
        f_out.close()

main()