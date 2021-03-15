import os
from pathlib import Path
import time
import click

def add_file_suffix(file_path, file_suffix):
    if len(file_path.split('.')) == 2:
        file_path = file_path.split('.')[0] + file_suffix + '.' + file_path.split('.')[-1]
    else:
        raise NotImplementedError
    return file_path

def timer(x, *args, **kwargs):
    """
    Arguments
    ------
    x: callable 

    Returns  
    ------
    time in second
    """
    s = time.time()
    x(*args, **kwargs)
    f = time.time()
    return f-s

def searching_all_files(directory: Path):   
    file_list = [] # A list for storing files existing in directories
    for x in directory.iterdir():
        if x.is_file():
            file_list.append(x)#here should be appended
        else:
            file_list.extend(searching_all_files(directory/x))# need to be extended
    return file_list

# def create_dir_if_not_exist(directory):
#     # create directory if not alreayd exist
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# def create_dir_if_not_exist(directory):
#     # create directory if not alreayd exist
#     if not os.path.exists(directory):
#         os.makedirs(directory)

