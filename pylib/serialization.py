import json
import os
import pickle


def _check_ext(path, default_ext):
    name, ext = os.path.splitext(path)
    if ext == '':
        if default_ext[0] == '.':
            default_ext = default_ext[1:]
        path = name + '.' + default_ext
    return path


def save_json(path, obj, **kwargs):
    # default
    if 'indent' not in kwargs:
        kwargs['indent'] = 4
    if 'separators' not in kwargs:
        kwargs['separators'] = (',', ': ')

    path = _check_ext(path, 'json')

    # wrap json.dump
    with open(path, 'w') as f:
        json.dump(obj, f, **kwargs)


def load_json(path, **kwargs):
    # wrap json.load
    with open(path) as f:
        return json.load(f, **kwargs)


def save_yaml(path, data, **kwargs):
    import oyaml as yaml

    path = _check_ext(path, 'yml')

    with open(path, 'w') as f:
        yaml.dump(data, f, **kwargs)


def load_yaml(path, **kwargs):
    import oyaml as yaml
    with open(path) as f:
        return yaml.load(f, **kwargs)


def save_pickle(path, obj, **kwargs):

    path = _check_ext(path, 'pkl')

    # wrap pickle.dump
    with open(path, 'wb') as f:
        pickle.dump(obj, f, **kwargs)


def load_pickle(path, **kwargs):
    # wrap pickle.load
    with open(path, 'rb') as f:
        return pickle.load(f, **kwargs)
