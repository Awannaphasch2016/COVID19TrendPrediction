import boto3 
from pathlib import Path
from global_params import *

s3 = boto3.client('s3')

def save_to_s3(save_path, save_file):
    # Method 1: Object.put()
    file_dir = Path(save_path) / save_file
    if file_dir.is_file():
        s3 = boto3.resource('s3')
        # object = s3.Object('covid19trendprediction', )
        # object.put(Body=some_binary_data)
        # s3.Object('covid19trendprediction', 'random_stuff.txt').put(Body=open('Data/Scratches/random_stuff.txt', 'rb'))
        s3.Object('covid19trendprediction', save_file).put(Body=open(str(file_dir), 'rb'))
    else:
        raise ValueError(f"{file_dir} doesn't exists. \n Premature termination.")

if __name__ == '__main__':
    # binary_data = b'randome text'
    BASEPATH = '/home/awannaphasch2016/Documents/Working/COVID19TrendPrediction/'
    save_path = BASEPATH + 'Data/Scratches'
    save_file = 'random_stuff.txt'
    save_to_s3(save_path, save_file)

