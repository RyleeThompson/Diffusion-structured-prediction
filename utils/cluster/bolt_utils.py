import turibolt as bolt
import os
import json


def get_save_path(task=None):
    if task is not None:
        dir = bolt.ARTIFACT_DIR + f'/info/{task.id}'
    else:
        dir = bolt.ARTIFACT_DIR + \
            '/info/{}'.format(bolt.get_current_task_id())

    os.makedirs(dir, exist_ok=True)
    return dir


def get_results_dir(task=None):
    if task is not None:
        dir = bolt.ARTIFACT_DIR + f'/results/{task.id}'
    else:
        dir = bolt.ARTIFACT_DIR + \
            '/results/{}'.format(bolt.get_current_task_id())

    os.makedirs(dir, exist_ok=True)
    return dir


def download_config(task):
    dl_path = get_save_path(task)
    if not os.path.exists(dl_path + '/config.json'):
        print('Downloading config')
        try:
            src_path = 'config.json'
            task.artifacts.download_file(
                src=src_path, dest=dl_path + '/config.json')
        except:
            src_path = f'info/{task.id}/config.json'
            task.artifacts.download_file(
                src=src_path, dest=dl_path + '/config.json')

    cfg = json.load(open(dl_path + '/config.json', 'r'))
    return cfg


def get_task_list(task_ids):
    tasks = []
    for id in task_ids:
        task = bolt.get_task(id)
        if task.children_count == 0:
            tasks.append(task)
        else:
            tasks += list(task.children)
    return tasks
