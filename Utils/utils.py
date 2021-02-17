import os
from pathlib import Path

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
