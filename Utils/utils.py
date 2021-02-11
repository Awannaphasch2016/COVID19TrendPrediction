import os


def create_dir_if_not_exist(directory):
    # create directory if not alreayd exist
    if not os.path.exists(directory):
        os.makedirs(directory)
