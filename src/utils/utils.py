import os


def get_file_from_src(folder, filename):
    if folder is not None:
        return os.path.join(os.path.dirname(__file__), f'../{folder}', filename)
    

