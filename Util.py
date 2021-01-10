import os

def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def normalize(weights):
    denom = sum(weights)
    if denom > 0.:
        return [float(i) / denom for i in weights]
    else:
        return [0.] * len(weights)