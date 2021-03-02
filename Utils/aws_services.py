import boto3 
from pathlib import Path
from global_params import *

s3 = boto3.client('s3')

def save_to_s3(save_path):
    # Method 1: Object.put()
    save_path = Path(save_path)
    file_dir = '/'.join(str(save_path.parents[0]).split(PROJECT_PATH)[-1].split('/')[1:])
    save_path = Path(file_dir) / save_path.name
    # print(save_path)
    if save_path.is_file():
        # print(file_dir)
        # print(save_path)
        s3 = boto3.resource('s3')
        # object = s3.Object('covid19trendprediction', )
        # object.put(Body=some_binary_data)
        # s3.Object('covid19trendprediction', 'random_stuff.txt').put(Body=open('Data/Scratches/random_stuff.txt', 'rb'))
        s3.Object('covid19trendprediction', str(save_path)).put(Body=open(str(save_path), 'rb'))
    else:
        raise ValueError(f"{save_path} doesn't exists. \n Premature termination.")

if __name__ == '__main__':
    # binary_data = b'randome text'
    BASEPATH = '/home/awannaphasch2016/Documents/Working/COVID19TrendPrediction/'
    save_path = BASEPATH + 'Data/Scratches'
    save_file = 'random_stuff.txt'
    save_to_s3(save_path, save_file)

