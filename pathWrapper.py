import os
import re
import glob


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_file_paths(root, file_type="/"):
    paths = sorted(glob.glob(f'{root}/*{file_type}'), key=natural_keys)
    return paths

def mk_dir(paths):
    if isinstance(paths, list):
        for path in paths:
            index = path.rfind('/')
            if "." in path[index:]:
                os.makedirs(path[:index], exist_ok=True)
            else:
                os.makedirs(path, exist_ok=True)
    elif isinstance(paths, str):
        index = paths.rfind('/')
        if "." in paths[index:]:
            os.makedirs(paths[:index], exist_ok=True)
        else:
            os.makedirs(paths, exist_ok=True)
