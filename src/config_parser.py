import os
import yaml
import glob
import shutil
import easydict
repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_config_from_yaml(yaml_file):
    """
    Get the config from a json file
    :param yaml_file:
    :return: config(namespace) or config(dictionary)
    """
    # Parse the configurations from the config yaml file provided
    with open(yaml_file, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)

    return config_dict


def merge_dicts(base_dict, other_dict):
    """ Merge two dicts
    Ensure that the base_dict remains as is and overwrite info from other_dict
    """
    if other_dict is None:
        return base_dict
    base_dict_type = type(base_dict)
    other_dict_type = type(other_dict)
    if other_dict_type != base_dict_type:
        pass
    if not issubclass(base_dict_type, dict):
        return other_dict
    return {k: merge_dicts(v, other_dict.get(k)) for k, v in base_dict.items()}


def process_config(config_file, default_config_file):
    """merge config between `default_config` and `your_config`
    """
    config_dict = get_config_from_yaml(config_file)
    default_config_dict = get_config_from_yaml(os.path.join(repo_dir, default_config_file))

    # Merge default and current config
    config_dict = merge_dicts(default_config_dict, config_dict)

    # Convert the dictionary to a namespace using EasyDict lib
    config = easydict.EasyDict(config_dict)
    config.config_path = os.path.realpath(config_file)
    if not config.project_name:
        config.project_name = '.'.join(config_file.split('/')[-1].split('.')[:-1])
    config.checkpoint_dir = config.checkpoint_path if config.checkpoint_path \
        else os.path.join('experiments', config.project_name)

    return config


def create_ckpt_dir(config):
    """Create checkpoint directory
    """
    # if checkpoint directory doesnt exists, create
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
        shutil.copy(config.config_path, os.path.join(config.checkpoint_dir, 'config_0.yaml'))
        return

    # if checkpoint directory exists, check if configs are the same
    config_fns = sorted(glob.glob(os.path.join(config.checkpoint_dir, 'config_*.yaml')))
    previous_idx = -1
    previous_config_fn = None
    for config_fn in config_fns:
        _idx = int(config_fn.split('/')[-1].split('_')[1].split('.')[0])
        if _idx > previous_idx:
            previous_idx = _idx
            previous_config_fn = config_fn
    if previous_config_fn:  # if exists
        previous_config_dict = get_config_from_yaml(previous_config_fn)
        config_dict = get_config_from_yaml(config.config_path)
        if previous_config_dict != config_dict:
            print('[WARN] your are using a different config setting from previous, please be awared')
        else:  # if configs are the same, keep as is
            return
    shutil.copy(
        config.config_path,
        os.path.join(config.checkpoint_dir, 'config_%d.yaml' % (previous_idx+1))
    )